"""
NoSQL Database + XML Storage
-----------------------------
End-Semester Enhancement: Flexible schema for heterogeneous rehabilitation
session data, stored in TinyDB (file-based NoSQL) with XML export capability.

Schema matches research paper specification (PPT Slide 10):
    <session>
      <user_id>, <timestamp>, <exercise>, <frames>,
      <correctness>, <confidence>, <compensation_detected>,
      <fluidity_score>
    </session>
"""

import os
import json
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
from typing import Dict, List, Optional, Any

# Use TinyDB if available, otherwise fall back to plain-JSON file store
try:
    from tinydb import TinyDB, Query
    TINYDB_AVAILABLE = True
except ImportError:
    TINYDB_AVAILABLE = False


# ------------------------------------------------------------------
# Fallback: Simple JSON file store (no extra dependency)
# ------------------------------------------------------------------

class _JsonStore:
    """Minimal dict-based store backed by a single JSON file."""

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, 'r') as f:
                self._data: Dict[str, dict] = json.load(f)
        else:
            self._data = {}

    def insert(self, record: dict) -> str:
        key = record['session_id']
        self._data[key] = record
        self._save()
        return key

    def get(self, session_id: str) -> Optional[dict]:
        return self._data.get(session_id)

    def search(self, **filters) -> List[dict]:
        results = []
        for rec in self._data.values():
            if all(rec.get(k) == v for k, v in filters.items()):
                results.append(rec)
        return results

    def all(self) -> List[dict]:
        return list(self._data.values())

    def count(self) -> int:
        return len(self._data)

    def _save(self):
        with open(self._path, 'w') as f:
            json.dump(self._data, f, indent=2, default=str)


# ------------------------------------------------------------------
# RehabDatabase
# ------------------------------------------------------------------

class RehabDatabase:
    """
    NoSQL database for rehabilitation session data.

    Supports:
      - Session save / retrieve
      - Patient history queries
      - Exercise-level statistics
      - Population benchmark computation
      - XML export (per-session or bulk)

    Parameters
    ----------
    db_path : str
        Directory to store database files (default 'rehab_data/').
    """

    def __init__(self, db_path: str = "rehab_data"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        sessions_file = os.path.join(db_path, "sessions.json")

        if TINYDB_AVAILABLE:
            self._db = TinyDB(sessions_file)
            self._table = self._db.table('sessions')
            self._backend = 'tinydb'
        else:
            self._store = _JsonStore(sessions_file)
            self._backend = 'json'

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_session(self, session: Dict) -> str:
        """
        Persist a session record and return its session_id.

        Expected keys (non-mandatory fields will default):
            patient_id, exercise_id, repetition_id (optional),
            timestamp (ISO 8601), correctness (0/1),
            confidence (float), rmse (float),
            compensation_found (bool), compensation_types (list),
            fluidity_score (float), fluidity_interpretation (str),
            frame_data (list of dicts) — optional raw joint data
        """
        session_id = session.setdefault('session_id', str(uuid.uuid4()))
        session.setdefault('timestamp', datetime.now().isoformat())
        session.setdefault('patient_id', 'unknown')
        session.setdefault('exercise_id', 'unknown')
        session.setdefault('correctness', None)
        session.setdefault('confidence', None)
        session.setdefault('rmse', None)
        session.setdefault('compensation_found', False)
        session.setdefault('compensation_types', [])
        session.setdefault('fluidity_score', None)
        session.setdefault('fluidity_interpretation', '')
        session.setdefault('frame_data', [])

        if self._backend == 'tinydb':
            self._table.insert(session)
        else:
            self._store.insert(session)

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a session by ID."""
        if self._backend == 'tinydb':
            Q = Query()
            results = self._table.search(Q.session_id == session_id)
            return results[0] if results else None
        else:
            return self._store.get(session_id)

    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """All sessions for a specific patient, newest first."""
        if self._backend == 'tinydb':
            Q = Query()
            results = self._table.search(Q.patient_id == patient_id)
        else:
            results = self._store.search(patient_id=patient_id)

        return sorted(results, key=lambda r: r.get('timestamp', ''), reverse=True)

    def get_exercise_sessions(self, exercise_id: str) -> List[Dict]:
        """All sessions for a specific exercise."""
        exid = exercise_id.replace('Ex', '').strip()
        if self._backend == 'tinydb':
            Q = Query()
            return self._table.search(Q.exercise_id == exercise_id)
        else:
            all_sessions = self._store.all()
            return [
                s for s in all_sessions
                if str(s.get('exercise_id', '')).replace('Ex', '').strip() == exid
            ]

    def get_all_sessions(self) -> List[Dict]:
        """Return every session in the database."""
        if self._backend == 'tinydb':
            return self._table.all()
        else:
            return self._store.all()

    def total_sessions(self) -> int:
        if self._backend == 'tinydb':
            return len(self._table)
        else:
            return self._store.count()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_patient_stats(self, patient_id: str) -> Dict:
        """Per-patient aggregated statistics."""
        history = self.get_patient_history(patient_id)
        if not history:
            return {'patient_id': patient_id, 'sessions': 0}

        correctness_vals = [s['correctness'] for s in history if s.get('correctness') is not None]
        fluidity_vals = [s['fluidity_score'] for s in history if s.get('fluidity_score') is not None]
        rmse_vals = [s['rmse'] for s in history if s.get('rmse') is not None]
        comp_counts = sum(1 for s in history if s.get('compensation_found'))

        return {
            'patient_id': patient_id,
            'total_sessions': len(history),
            'accuracy_rate': float(sum(correctness_vals) / len(correctness_vals)) if correctness_vals else None,
            'mean_fluidity': float(sum(fluidity_vals) / len(fluidity_vals)) if fluidity_vals else None,
            'mean_rmse': float(sum(rmse_vals) / len(rmse_vals)) if rmse_vals else None,
            'compensation_rate': comp_counts / len(history),
            'last_session': history[0]['timestamp'] if history else None,
        }

    def get_exercise_stats(self, exercise_id: str) -> Dict:
        """Aggregate statistics for a specific exercise across all patients."""
        sessions = self.get_exercise_sessions(exercise_id)
        if not sessions:
            return {'exercise_id': exercise_id, 'sessions': 0}

        correctness_vals = [s['correctness'] for s in sessions if s.get('correctness') is not None]
        rmse_vals = [s['rmse'] for s in sessions if s.get('rmse') is not None]
        fluidity_vals = [s['fluidity_score'] for s in sessions if s.get('fluidity_score') is not None]

        return {
            'exercise_id': exercise_id,
            'total_sessions': len(sessions),
            'accuracy_rate': float(sum(correctness_vals) / len(correctness_vals)) if correctness_vals else None,
            'mean_rmse': float(sum(rmse_vals) / len(rmse_vals)) if rmse_vals else None,
            'std_rmse': float(float(sum((v - sum(rmse_vals)/len(rmse_vals))**2 for v in rmse_vals) / len(rmse_vals)) ** 0.5) if len(rmse_vals) > 1 else 0.0,
            'mean_fluidity': float(sum(fluidity_vals) / len(fluidity_vals)) if fluidity_vals else None,
        }

    def get_population_benchmark(self, exercise_id: str) -> Dict:
        """
        Population-level benchmark stats for one exercise.
        Includes RMSE percentiles used for patient comparison.
        """
        sessions = self.get_exercise_sessions(exercise_id)
        rmse_vals = [s['rmse'] for s in sessions if s.get('rmse') is not None]

        if not rmse_vals:
            return {'exercise_id': exercise_id, 'n': 0}

        rmse_arr = sorted(rmse_vals)
        n = len(rmse_arr)

        def pct(p):
            idx = max(0, min(n - 1, int(p / 100 * n)))
            return rmse_arr[idx]

        return {
            'exercise_id': exercise_id,
            'n': n,
            'mean_rmse': float(sum(rmse_arr) / n),
            'std_rmse':  float((sum((v - sum(rmse_arr)/n)**2 for v in rmse_arr) / n) ** 0.5),
            'percentiles': {
                'p10': pct(10), 'p25': pct(25), 'p50': pct(50),
                'p75': pct(75), 'p90': pct(90),
            },
            'accuracy_rate': self.get_exercise_stats(exercise_id).get('accuracy_rate'),
        }

    # ------------------------------------------------------------------
    # XML Export
    # ------------------------------------------------------------------

    def export_session_xml(self, session_id: str) -> str:
        """
        Export a single session as XML string matching research paper schema.

        Returns
        -------
        str : Pretty-printed XML string.
        """
        session = self.get_session(session_id)
        if session is None:
            return f"<error>Session {session_id} not found</error>"

        return self._session_to_xml(session)

    def export_all_xml(self, output_path: Optional[str] = None) -> str:
        """Export all sessions to XML."""
        root = ET.Element("rehab_database")
        root.set("exported_at", datetime.now().isoformat())
        root.set("total_sessions", str(self.total_sessions()))

        for session in self.get_all_sessions():
            session_el = self._build_xml_element(session)
            root.append(session_el)

        xml_str = self._prettify(root)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)

        return xml_str

    @staticmethod
    def _session_to_xml(session: Dict) -> str:
        root = RehabDatabase._build_xml_element(session)
        return RehabDatabase._prettify(root)

    @staticmethod
    def _build_xml_element(session: Dict) -> ET.Element:
        """Convert session dict to XML Element as per research paper schema."""
        el = ET.Element("session")
        el.set("id", str(session.get('session_id', '')))

        def add(tag, val):
            child = ET.SubElement(el, tag)
            child.text = str(val) if val is not None else ""

        add("user_id", session.get('patient_id', ''))
        add("timestamp", session.get('timestamp', ''))
        add("exercise", session.get('exercise_id', ''))
        add("correctness", session.get('correctness', ''))
        add("confidence", session.get('confidence', ''))
        add("rmse", session.get('rmse', ''))
        add("compensation_detected", str(session.get('compensation_found', False)).lower())
        comp_types = session.get('compensation_types', [])
        add("compensation_types", ", ".join(comp_types) if comp_types else "none")
        add("fluidity_score", session.get('fluidity_score', ''))
        add("fluidity_interpretation", session.get('fluidity_interpretation', ''))

        # Optional frame data
        frame_data = session.get('frame_data', [])
        if frame_data:
            frames_el = ET.SubElement(el, "frames")
            frames_el.set("count", str(len(frame_data)))
            for i, frame in enumerate(frame_data[:5]):  # limit to first 5 frames in XML
                frame_el = ET.SubElement(frames_el, "frame")
                frame_el.set("id", str(i))
                if isinstance(frame, dict):
                    for k, v in frame.items():
                        add_f = ET.SubElement(frame_el, k)
                        add_f.text = str(v)

        return el

    @staticmethod
    def _prettify(element: ET.Element) -> str:
        rough = ET.tostring(element, encoding='unicode')
        reparsed = minidom.parseString(rough)
        return reparsed.toprettyxml(indent="  ")


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import shutil

    print("Testing RehabDatabase...")
    test_dir = "test_rehab_db"

    db = RehabDatabase(db_path=test_dir)

    # Insert test sessions
    s1_id = db.save_session({
        'patient_id': 'patient_001',
        'exercise_id': 'Ex1',
        'correctness': 1,
        'confidence': 0.87,
        'rmse': 0.24,
        'compensation_found': False,
        'compensation_types': [],
        'fluidity_score': 0.75,
        'fluidity_interpretation': 'Good',
    })

    s2_id = db.save_session({
        'patient_id': 'patient_001',
        'exercise_id': 'Ex1',
        'correctness': 0,
        'confidence': 0.72,
        'rmse': 0.45,
        'compensation_found': True,
        'compensation_types': ['trunk_lean'],
        'fluidity_score': 0.52,
        'fluidity_interpretation': 'Fair',
    })

    print(f"\n  Session 1 ID: {s1_id}")
    print(f"  Session 2 ID: {s2_id}")
    print(f"  Total sessions: {db.total_sessions()}")

    stats = db.get_patient_stats('patient_001')
    print(f"\n  Patient stats: {stats}")

    xml_out = db.export_session_xml(s1_id)
    print(f"\n  Session XML:\n{xml_out}")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    print("  [OK] Database module OK")
