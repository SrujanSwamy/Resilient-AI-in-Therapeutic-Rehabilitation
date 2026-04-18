"""
Cloud Server Launcher
----------------------
One-command startup for the RehabAI Cloud Stack.

Usage:
    python run_cloud.py                  # Start dashboard on port 5000
    python run_cloud.py --api            # Start FastAPI on port 8000
    python run_cloud.py --both           # Start both (dashboard + API)
    python run_cloud.py --demo Ex1       # Run end-to-end demo then launch dashboard
    python run_cloud.py --seed           # Seed DB with synthetic data for demo
"""

import os
import sys
import argparse
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'subteam1_edge'))


def seed_database(db_path: str = "rehab_data", n: int = 60):
    """
    Insert synthetic session data so the dashboard has something to show.
    """
    import random
    from subteam2_cloud.database import RehabDatabase

    random.seed(99)
    db = RehabDatabase(db_path=db_path)

    patients = [f"patient_{i:03d}" for i in range(1, 6)]
    exercises = ["Ex1", "Ex2", "Ex3", "Ex4", "Ex5", "Ex6"]
    comp_types_pool = ["trunk_lean", "hip_shift", "shoulder_shrug"]

    print(f"Seeding database with {n} synthetic sessions...")
    from datetime import datetime, timedelta
    base_time = datetime(2026, 3, 1, 9, 0, 0)

    for i in range(n):
        pid = random.choice(patients)
        eid = random.choice(exercises)
        has_comp = random.random() < 0.25
        db.save_session({
            "patient_id": pid,
            "exercise_id": eid,
            "correctness": random.choice([1, 1, 1, 0]),
            "confidence": round(random.uniform(0.62, 0.99), 3),
            "rmse": round(random.uniform(0.12, 0.52), 4),
            "compensation_found": has_comp,
            "compensation_types": random.sample(comp_types_pool, k=random.randint(1,2)) if has_comp else [],
            "compensation_severity": round(random.uniform(0.3, 0.9), 2) if has_comp else 0.0,
            "fluidity_score": round(random.uniform(0.3, 0.92), 3),
            "fluidity_interpretation": random.choice([
                "Excellent — smooth, coordinated movement",
                "Good — minor hesitations detected",
                "Fair — noticeable jerkiness, improvement needed",
            ]),
            "prediction_method": random.choice(["CNN", "Threshold"]),
            "timestamp": (base_time + timedelta(days=i, hours=random.randint(0, 8))).isoformat(),
        })

    print(f"  [OK] Seeded {n} sessions. DB total: {db.total_sessions()}")


def start_dashboard(port: int = 5000):
    """Launch Flask clinician dashboard."""
    try:
        from subteam2_cloud.dashboard import run_dashboard
        run_dashboard(host="0.0.0.0", port=port, debug=False)
    except ImportError as e:
        print(f"[Dashboard] Import error: {e}")
        print("  Install with: pip install flask")


def start_api(port: int = 8000):
    """Launch FastAPI REST API."""
    try:
        import uvicorn
        print(f"\nStarting FastAPI REST API on http://localhost:{port}")
        print(f"API docs: http://localhost:{port}/docs\n")
        uvicorn.run(
            "subteam2_cloud.api:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="warning",
        )
    except ImportError:
        print("[API] uvicorn not installed. Install with: pip install uvicorn")


def run_e2e_demo(exercise_id: str = "Ex1", dataset_path: str = "dataset"):
    """Run end-to-end pipeline demo."""
    from subteam2_cloud.pipeline_integration import run_full_pipeline
    run_full_pipeline(
        exercise_id=exercise_id,
        dataset_path=dataset_path,
        patient_id="patient_demo",
        db_path="rehab_data",
        n_demo_samples=5,
        train_classifier=True,
    )


def run_live_webcam(exercise_id: str = "Ex1", dataset_path: str = "dataset",
                   camera: int = 0, patient: str = "webcam_patient",
                   no_movenet: bool = False):
    """Launch live webcam exercise feedback."""
    sys.path.insert(0, os.path.join(ROOT, 'subteam1_edge'))
    from live_webcam import run_live
    run_live(
        exercise_id=exercise_id,
        dataset_path=dataset_path,
        camera_idx=camera,
        use_movenet=not no_movenet,
        patient_id=patient,
        db_path="rehab_data",
    )


def main():
    parser = argparse.ArgumentParser(
        description="RehabAI Cloud Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cloud.py                    # Dashboard at http://localhost:5000
  python run_cloud.py --api              # FastAPI at http://localhost:8000
  python run_cloud.py --both             # Both servers
  python run_cloud.py --seed             # Seed DB with demo data, then start
  python run_cloud.py --demo Ex1         # Run e2e pipeline then dashboard
  python run_cloud.py --demo Ex1 --api   # e2e demo + FastAPI
        """,
    )
    parser.add_argument("--api", action="store_true", help="Start FastAPI REST API")
    parser.add_argument("--both", action="store_true", help="Start dashboard + API")
    parser.add_argument("--demo", metavar="EXERCISE", help="Run end-to-end demo (e.g. Ex1)")
    parser.add_argument("--live", metavar="EXERCISE",
                        help="Start live webcam mode (e.g. --live Ex1)")
    parser.add_argument("--dataset", default="dataset", help="Dataset path for demo")
    parser.add_argument("--seed", action="store_true", help="Seed DB with synthetic demo data")
    parser.add_argument("--dash-port", type=int, default=5000)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--no-dash", action="store_true", help="Do not start dashboard")
    parser.add_argument("--camera", type=int, default=0, help="Webcam camera index")
    parser.add_argument("--no-movenet", action="store_true",
                        help="Disable MoveNet in webcam mode (synthetic skeleton)")
    parser.add_argument("--patient", default="webcam_patient",
                        help="Patient ID for webcam session")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RehabAI Cloud Server")
    print("  Resilient AI in Therapeutic Rehabilitation")
    print("="*60)

    # Seed database if requested
    if args.seed:
        seed_database()

    # End-to-end demo
    if args.demo:
        print(f"\nRunning end-to-end pipeline demo for {args.demo}...")
        try:
            run_e2e_demo(exercise_id=args.demo, dataset_path=args.dataset)
        except Exception as e:
            print(f"  [!] Demo error: {e}")
            import traceback
            traceback.print_exc()

    # Live webcam mode
    if args.live:
        print(f"\nStarting live webcam for {args.live}...")
        try:
            run_live_webcam(
                exercise_id=args.live,
                dataset_path=args.dataset,
                camera=args.camera,
                patient=args.patient,
                no_movenet=args.no_movenet,
            )
        except Exception as e:
            print(f"  [!] Live webcam error: {e}")
            import traceback
            traceback.print_exc()
        return

    # Start servers
    start_api_flag = args.api or args.both
    start_dash_flag = not args.no_dash and (not args.api or args.both)

    threads = []

    if start_api_flag:
        t = threading.Thread(target=start_api, args=(args.api_port,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)

    if start_dash_flag:
        print(f"\nStarting Clinician Dashboard at http://localhost:{args.dash_port}")
        print(f"  Dashboard URL: http://localhost:{args.dash_port}")
        if start_api_flag:
            print(f"  REST API URL:  http://localhost:{args.api_port}/docs")
        start_dashboard(port=args.dash_port)
    elif start_api_flag:
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
