"""
REST API — Rehabilitation Cloud Server
---------------------------------------
End-Semester Enhancement: FastAPI REST API for session upload, retrieval,
analytics, XML export and population benchmarking.

Endpoints:
    GET  /health
    POST /sessions                          Upload session from edge
    GET  /sessions/{session_id}             Get session by ID
    GET  /sessions/{session_id}/xml         Export session as XML
    GET  /patients/{patient_id}/history     Patient history
    GET  /patients/{patient_id}/report      Full patient benchmark report
    GET  /exercises/{exercise_id}/stats     Exercise aggregate stats
    GET  /benchmark/{exercise_id}           Population benchmark
    GET  /sessions                          List all sessions (paginated)

Run with:
    uvicorn subteam2_cloud.api:app --reload --port 8000
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from subteam2_cloud.database import RehabDatabase
from subteam2_cloud.benchmarking import PopulationBenchmark

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# FastAPI setup (with graceful fallback if not installed)
# -----------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, Response, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[API] FastAPI not installed. Install with: pip install fastapi uvicorn")

# -----------------------------------------------------------------------
# Shared database instance
# -----------------------------------------------------------------------

DB_PATH = os.path.join(ROOT, "rehab_data")
_db = RehabDatabase(db_path=DB_PATH)
_bench = PopulationBenchmark(_db)

# -----------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class SessionUpload(BaseModel):
        patient_id: str = Field(..., example="patient_001")
        exercise_id: str = Field(..., example="Ex1")
        correctness: Optional[int] = Field(None, ge=0, le=1, example=1)
        confidence: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.87)
        rmse: Optional[float] = Field(None, ge=0.0, example=0.24)
        compensation_found: bool = False
        compensation_types: List[str] = []
        fluidity_score: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.75)
        fluidity_interpretation: str = ""
        prediction_method: str = "Threshold"
        frame_data: List[Dict] = []
        timestamp: Optional[str] = None

    class SessionResponse(BaseModel):
        session_id: str
        message: str
        timestamp: str

# -----------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="RehabAI Cloud API",
        description=(
            "REST API for the Resilient AI in Therapeutic Rehabilitation system.\n\n"
            "Provides session upload, patient history, exercise analytics, "
            "population benchmarking, and XML export."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    @app.get("/health", tags=["System"])
    def health():
        """Service health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_sessions": _db.total_sessions(),
            "db_backend": _db._backend,
        }

    # -------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------

    @app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
    def upload_session(session: SessionUpload):
        """
        Upload a completed exercise session from the edge.
        Returns the assigned session_id.
        """
        data = session.dict()
        data["timestamp"] = data.get("timestamp") or datetime.now().isoformat()
        session_id = _db.save_session(data)
        return {
            "session_id": session_id,
            "message": "Session stored successfully",
            "timestamp": data["timestamp"],
        }

    @app.get("/sessions", tags=["Sessions"])
    def list_sessions(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        """List all sessions with pagination."""
        all_s = _db.get_all_sessions()
        all_s.sort(key=lambda s: s.get('timestamp', ''), reverse=True)
        return {
            "total": len(all_s),
            "offset": offset,
            "limit": limit,
            "sessions": all_s[offset: offset + limit],
        }

    @app.get("/sessions/{session_id}", tags=["Sessions"])
    def get_session(session_id: str):
        """Retrieve a session by ID."""
        session = _db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return session

    @app.get("/sessions/{session_id}/xml", tags=["Sessions"])
    def get_session_xml(session_id: str):
        """Export session as XML (research paper format)."""
        xml_str = _db.export_session_xml(session_id)
        if "not found" in xml_str:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return Response(content=xml_str, media_type="application/xml")

    @app.delete("/sessions/{session_id}", tags=["Sessions"])
    def delete_session(session_id: str):
        """Delete a session (for data management)."""
        session = _db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        # TinyDB removal
        if _db._backend == 'tinydb':
            from tinydb import Query
            Q = Query()
            _db._table.remove(Q.session_id == session_id)
        else:
            _db._store._data.pop(session_id, None)
            _db._store._save()
        return {"message": f"Session {session_id} deleted"}

    # -------------------------------------------------------------------
    # Patients
    # -------------------------------------------------------------------

    @app.get("/patients/{patient_id}/history", tags=["Patients"])
    def patient_history(patient_id: str):
        """All sessions for a patient, newest first."""
        history = _db.get_patient_history(patient_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"No sessions for patient {patient_id}")
        return {
            "patient_id": patient_id,
            "total_sessions": len(history),
            "sessions": history,
        }

    @app.get("/patients/{patient_id}/stats", tags=["Patients"])
    def patient_stats(patient_id: str):
        """Aggregated stats for a patient."""
        stats = _db.get_patient_stats(patient_id)
        if stats.get('total_sessions', 0) == 0:
            raise HTTPException(status_code=404, detail=f"No sessions for patient {patient_id}")
        return stats

    @app.get("/patients/{patient_id}/report", tags=["Patients"])
    def patient_report(patient_id: str):
        """Full benchmark report for a patient."""
        report = _bench.generate_patient_report(patient_id)
        if report['summary'].get('total_sessions', 0) == 0:
            raise HTTPException(status_code=404, detail=f"No sessions for patient {patient_id}")
        return report

    @app.get("/patients/{patient_id}/progress/{exercise_id}", tags=["Patients"])
    def patient_progress(patient_id: str, exercise_id: str, metric: str = "rmse"):
        """Time-series progress for a patient on a specific exercise."""
        progress = _bench.get_patient_progress(patient_id, exercise_id, metric=metric)
        return {
            "patient_id": patient_id,
            "exercise_id": exercise_id,
            "metric": metric,
            "data_points": progress,
        }

    # -------------------------------------------------------------------
    # Exercises
    # -------------------------------------------------------------------

    @app.get("/exercises/{exercise_id}/stats", tags=["Exercises"])
    def exercise_stats(exercise_id: str):
        """Aggregate statistics for a specific exercise across all patients."""
        stats = _db.get_exercise_stats(exercise_id)
        return stats

    @app.get("/exercises/{exercise_id}/sessions", tags=["Exercises"])
    def exercise_sessions(exercise_id: str):
        """All sessions for an exercise."""
        sessions = _db.get_exercise_sessions(exercise_id)
        return {
            "exercise_id": exercise_id,
            "total": len(sessions),
            "sessions": sessions,
        }

    # -------------------------------------------------------------------
    # Benchmarking
    # -------------------------------------------------------------------

    @app.get("/benchmark/{exercise_id}", tags=["Benchmarking"])
    def population_benchmark(exercise_id: str):
        """Population-level RMSE and accuracy statistics for an exercise."""
        stats = _bench.compute_population_stats(exercise_id)
        return stats

    @app.get("/benchmark/{exercise_id}/patient/{patient_id}", tags=["Benchmarking"])
    def patient_percentile(
        exercise_id: str,
        patient_id: str,
        metric: str = "rmse",
    ):
        """Patient's percentile rank within the population for given metric."""
        pct = _bench.get_patient_percentile(patient_id, exercise_id, metric=metric)
        if pct is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data for patient={patient_id} exercise={exercise_id}",
            )
        return {
            "patient_id": patient_id,
            "exercise_id": exercise_id,
            "metric": metric,
            "percentile": pct,
            "interpretation": (
                f"Patient performs better than {pct:.0f}% of the population"
                if metric == 'rmse' else
                f"Patient's {metric} is at the {pct:.0f}th percentile"
            ),
        }

    @app.get("/export/xml", tags=["Export"])
    def export_all_xml():
        """Export all sessions as a single XML document."""
        xml_str = _db.export_all_xml()
        return Response(content=xml_str, media_type="application/xml")


# -----------------------------------------------------------------------
# Main (for direct run without uvicorn)
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    try:
        import uvicorn
        print("Starting RehabAI Cloud API on http://localhost:8000")
        print("API docs available at: http://localhost:8000/docs")
        uvicorn.run("subteam2_cloud.api:app", host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
