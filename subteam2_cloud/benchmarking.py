"""
Population Benchmarking Module
-------------------------------
End-Semester Enhancement: Compare individual patient performance against
population-level statistics across all stored sessions.

Provides:
  - Population RMSE / accuracy / fluidity distributions
  - Patient percentile ranking
  - Progress tracking over sessions
  - Per-exercise trend analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database import RehabDatabase


class PopulationBenchmark:
    """
    Population-level benchmarking against stored session data.

    Parameters
    ----------
    db : RehabDatabase
    """

    def __init__(self, db: RehabDatabase):
        self.db = db

    # ------------------------------------------------------------------
    # Population statistics
    # ------------------------------------------------------------------

    def compute_population_stats(self, exercise_id: str) -> Dict:
        """
        Full population statistics for one exercise type.

        Returns
        -------
        dict with keys:
            n, mean_rmse, std_rmse, percentiles (dict),
            accuracy_rate, mean_fluidity,
            compensation_rate
        """
        sessions = self.db.get_exercise_sessions(exercise_id)
        if not sessions:
            return {'exercise_id': exercise_id, 'n': 0, 'message': 'No data'}

        rmse_vals      = [s['rmse'] for s in sessions if s.get('rmse') is not None]
        fluidity_vals  = [s['fluidity_score'] for s in sessions if s.get('fluidity_score') is not None]
        correct_vals   = [s['correctness'] for s in sessions if s.get('correctness') is not None]
        comp_count     = sum(1 for s in sessions if s.get('compensation_found'))

        def percentile(arr, p):
            if not arr:
                return None
            arr_s = sorted(arr)
            idx = max(0, min(len(arr_s) - 1, int(p / 100 * len(arr_s))))
            return arr_s[idx]

        result = {
            'exercise_id': exercise_id,
            'n': len(sessions),
        }

        if rmse_vals:
            mean_r = sum(rmse_vals) / len(rmse_vals)
            result['mean_rmse'] = mean_r
            result['std_rmse']  = (sum((v - mean_r) ** 2 for v in rmse_vals) / len(rmse_vals)) ** 0.5
            result['rmse_percentiles'] = {
                'p10': percentile(rmse_vals, 10),
                'p25': percentile(rmse_vals, 25),
                'p50': percentile(rmse_vals, 50),
                'p75': percentile(rmse_vals, 75),
                'p90': percentile(rmse_vals, 90),
            }

        if correct_vals:
            result['accuracy_rate'] = sum(correct_vals) / len(correct_vals)

        if fluidity_vals:
            result['mean_fluidity'] = sum(fluidity_vals) / len(fluidity_vals)

        result['compensation_rate'] = comp_count / len(sessions)
        return result

    # ------------------------------------------------------------------
    # Patient percentile
    # ------------------------------------------------------------------

    def get_patient_percentile(
        self,
        patient_id: str,
        exercise_id: str,
        metric: str = 'rmse',
    ) -> Optional[float]:
        """
        Compute which percentile the patient falls in for an exercise.
        Lower RMSE = better -> returns percentile of "lower than" fraction.

        Parameters
        ----------
        patient_id  : str
        exercise_id : str
        metric      : 'rmse' | 'fluidity_score' | 'correctness'

        Returns
        -------
        float in [0, 100] or None if no data.
        """
        all_sessions = self.db.get_exercise_sessions(exercise_id)
        all_vals = [s.get(metric) for s in all_sessions if s.get(metric) is not None]

        patient_sessions = [s for s in all_sessions if s.get('patient_id') == patient_id]
        patient_vals = [s.get(metric) for s in patient_sessions if s.get(metric) is not None]

        if not all_vals or not patient_vals:
            return None

        patient_avg = sum(patient_vals) / len(patient_vals)
        n_below = sum(1 for v in all_vals if v < patient_avg)
        percentile = (n_below / len(all_vals)) * 100.0

        # For RMSE, lower is better -> invert percentile
        if metric == 'rmse':
            percentile = 100.0 - percentile

        return round(percentile, 1)

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def get_patient_progress(
        self,
        patient_id: str,
        exercise_id: str,
        metric: str = 'rmse',
    ) -> List[Dict]:
        """
        Time-ordered metric values for a patient to track progress.

        Returns
        -------
        list of {timestamp, value, session_id}
        """
        history = self.db.get_patient_history(patient_id)
        exercise_history = [
            s for s in history
            if s.get('exercise_id') == exercise_id and s.get(metric) is not None
        ]
        exercise_history.sort(key=lambda s: s.get('timestamp', ''))
        return [
            {
                'timestamp': s.get('timestamp'),
                'value': s.get(metric),
                'session_id': s.get('session_id'),
                'correctness': s.get('correctness'),
            }
            for s in exercise_history
        ]

    # ------------------------------------------------------------------
    # Report generator
    # ------------------------------------------------------------------

    def generate_patient_report(self, patient_id: str) -> Dict:
        """
        Comprehensive patient performance report across all exercises.

        Returns
        -------
        dict with patient summary and per-exercise breakdown.
        """
        patient_stats = self.db.get_patient_stats(patient_id)
        all_sessions = self.db.get_patient_history(patient_id)

        exercise_ids = list({s.get('exercise_id') for s in all_sessions if s.get('exercise_id')})
        exercise_reports = {}

        for eid in exercise_ids:
            population_stats = self.compute_population_stats(eid)
            percentile_rmse = self.get_patient_percentile(patient_id, eid, 'rmse')
            percentile_fluidity = self.get_patient_percentile(patient_id, eid, 'fluidity_score')
            progress = self.get_patient_progress(patient_id, eid, 'rmse')

            patient_ex_sessions = [s for s in all_sessions if s.get('exercise_id') == eid]
            ex_rmse_vals = [s['rmse'] for s in patient_ex_sessions if s.get('rmse') is not None]
            ex_fluidity_vals = [s['fluidity_score'] for s in patient_ex_sessions if s.get('fluidity_score') is not None]
            ex_correct_vals = [s['correctness'] for s in patient_ex_sessions if s.get('correctness') is not None]

            exercise_reports[eid] = {
                'sessions': len(patient_ex_sessions),
                'patient_mean_rmse': sum(ex_rmse_vals) / len(ex_rmse_vals) if ex_rmse_vals else None,
                'patient_mean_fluidity': sum(ex_fluidity_vals) / len(ex_fluidity_vals) if ex_fluidity_vals else None,
                'patient_accuracy': sum(ex_correct_vals) / len(ex_correct_vals) if ex_correct_vals else None,
                'rmse_percentile': percentile_rmse,
                'fluidity_percentile': percentile_fluidity,
                'population_mean_rmse': population_stats.get('mean_rmse'),
                'progress': progress,
            }

        return {
            'patient_id': patient_id,
            'summary': patient_stats,
            'exercises': exercise_reports,
            'generated_at': __import__('datetime').datetime.now().isoformat(),
        }


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    print("Testing Population Benchmarking Module...")

    test_dir = "test_bench_db"
    db = RehabDatabase(db_path=test_dir)

    import random
    random.seed(42)

    # Populate synthetic data
    for i in range(30):
        pid = f"patient_{(i % 5) + 1:03d}"
        db.save_session({
            'patient_id': pid,
            'exercise_id': 'Ex1',
            'correctness': random.choice([0, 1]),
            'confidence': round(random.uniform(0.6, 0.98), 2),
            'rmse': round(random.uniform(0.15, 0.55), 3),
            'compensation_found': random.choice([True, False]),
            'compensation_types': ['trunk_lean'] if random.random() > 0.6 else [],
            'fluidity_score': round(random.uniform(0.3, 0.9), 2),
            'fluidity_interpretation': 'Good',
        })

    bench = PopulationBenchmark(db)

    pop_stats = bench.compute_population_stats('Ex1')
    print(f"\n  Population n={pop_stats['n']}")
    print(f"  Mean RMSE:     {pop_stats.get('mean_rmse', 0):.4f}")
    print(f"  Accuracy:      {pop_stats.get('accuracy_rate', 0):.2f}")
    print(f"  RMSE p50:      {pop_stats.get('rmse_percentiles', {}).get('p50', 0):.4f}")

    pct = bench.get_patient_percentile('patient_001', 'Ex1', 'rmse')
    print(f"\n  patient_001 RMSE percentile: {pct}")

    report = bench.generate_patient_report('patient_001')
    print(f"\n  Patient report exercises: {list(report['exercises'].keys())}")

    shutil.rmtree(test_dir, ignore_errors=True)
    print("  [OK] Benchmarking module OK")
