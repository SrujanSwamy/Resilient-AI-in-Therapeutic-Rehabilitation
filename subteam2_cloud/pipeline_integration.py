"""
End-to-End Pipeline Integration
---------------------------------
Ties together the Edge pipeline (Modules 1-5 + End-Sem enhancements)
with the Cloud components (PubSub, Database, Benchmarking).

This is the single entry-point for a complete end-to-end demonstration.

Usage:
    python subteam2_cloud/pipeline_integration.py --exercise Ex1

Or import and call:
    from subteam2_cloud.pipeline_integration import run_full_pipeline
    result = run_full_pipeline("Ex1", dataset_path="dataset")
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'subteam1_edge'))

# -----------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------
from subteam1_edge.normalization import SkeletonNormalizer
from subteam1_edge.reference_model import ReferenceModel
from subteam1_edge.comparison import MovementComparator
from subteam1_edge.confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD
from subteam1_edge.classifier import ExerciseClassifier
from subteam1_edge.imputation import JointImputer
from subteam1_edge.compensation import CompensationDetector
from subteam1_edge.fluidity import FluididtyAnalyzer

from subteam2_cloud.database import RehabDatabase
from subteam2_cloud.pubsub import MessageBroker, EdgePublisher, CloudSubscriber
from subteam2_cloud.benchmarking import PopulationBenchmark

try:
    from subteam1_edge.movenet import TF_AVAILABLE
except Exception:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Module instances (shared singletons)
# -----------------------------------------------------------------------
_imputer = JointImputer(confidence_threshold=0.3)
_compensation = CompensationDetector(severity_threshold=0.3)
_fluidity = FluididtyAnalyzer(target_frames=48)


def _print_section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# -----------------------------------------------------------------------
# Per-sample enhanced processing
# -----------------------------------------------------------------------

def process_sample_enhanced(
    sample_seq: "np.ndarray",        # (frames, 17, 2)  normalised
    raw_keypoints_seq: "Optional[np.ndarray]",   # (frames, 17, 3) with conf
    exercise_id: str,
    reference_mean: "Optional[np.ndarray]" = None,
) -> Dict:
    """
    Run all end-semester enhancements on a single sample.

    Parameters
    ----------
    sample_seq : np.ndarray  (frames, 17, 2)  — already normalised
    raw_keypoints_seq : np.ndarray or None    — (frames, 17, 3) with confidence
    exercise_id : str
    reference_mean : np.ndarray or None       — (frames, 17, 2) from Module 3

    Returns
    -------
    dict with compensation, fluidity keys ready to merge into session data.
    """
    import numpy as np

    enhancements: Dict = {}

    # --- Imputation report (applied to raw keypoints before normalisation) ---
    if raw_keypoints_seq is not None:
        _, imp_report = _imputer.impute_sequence(raw_keypoints_seq)
        enhancements['imputation'] = {
            'total_missing': imp_report['total_missing'],
            'temporal_filled': imp_report['temporal'],
            'spatial_filled': imp_report['spatial'],
            'motion_filled': imp_report['motion'],
            'unfilled': imp_report['unfilled'],
        }
    else:
        enhancements['imputation'] = None

    # --- Compensation detection ---
    comp_result = _compensation.detect(sample_seq, exercise_id, reference_mean)
    enhancements['compensation_found'] = comp_result['compensation_found']
    enhancements['compensation_types'] = comp_result['types']
    enhancements['compensation_severity'] = comp_result['severity']
    enhancements['compensation_details'] = comp_result['details']
    enhancements['compensation_feedback'] = _compensation.get_feedback_message(comp_result)

    # --- Fluidity analysis ---
    fluidity_result = _fluidity.analyze(sample_seq)
    enhancements['fluidity_score'] = fluidity_result['overall_fluidity']
    enhancements['fluidity_jerk'] = fluidity_result['jerk_score']
    enhancements['fluidity_velocity'] = fluidity_result['velocity_consistency']
    enhancements['fluidity_interpretation'] = fluidity_result['interpretation']

    return enhancements


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------

def run_full_pipeline(
    exercise_id: str = "Ex1",
    dataset_path: str = "dataset",
    patient_id: str = "patient_demo",
    db_path: str = "rehab_data",
    n_demo_samples: int = 5,
    train_classifier: bool = True,
) -> Dict:
    """
    Complete end-to-end pipeline:
      Edge: Modules 1-5 + Imputation + Compensation + Fluidity
      Cloud: PubSub -> Database -> Benchmarking

    Parameters
    ----------
    exercise_id : str   e.g. "Ex1"
    dataset_path : str  path to REHAB24-6 dataset
    patient_id : str    patient identifier for demo
    db_path : str       cloud database directory
    n_demo_samples : int  number of samples to run through full pipeline
    train_classifier : bool  whether to train classifier (takes time)

    Returns
    -------
    dict : pipeline summary
    """
    import numpy as np

    print("\n" + "="*70)
    print("  RESILIENT AI IN THERAPEUTIC REHABILITATION")
    print("  End-to-End Pipeline — End-Semester Implementation")
    print("="*70)
    print(f"  Exercise:       {exercise_id}")
    print(f"  Patient:        {patient_id}")
    print(f"  Dataset:        {dataset_path}")
    print(f"  TF Available:   {TF_AVAILABLE}")
    print("="*70)

    # ------------------------------------------------------------------
    # CLOUD SETUP
    # ------------------------------------------------------------------
    _print_section("CLOUD: Initialising Database & PubSub")

    db = RehabDatabase(db_path=db_path)
    broker = MessageBroker(mode="local")
    edge_pub = EdgePublisher(broker, edge_id="edge_01")
    cloud_sub = CloudSubscriber(broker, database=db)
    broker.start()
    bench = PopulationBenchmark(db)

    print(f"  [OK] Database ready  (sessions: {db.total_sessions()})")
    print(f"  [OK] Pub/Sub broker  started")

    # ------------------------------------------------------------------
    # EDGE: Module 3 — Reference Model
    # ------------------------------------------------------------------
    _print_section("EDGE: Module 3 — Building Reference Model")

    ref_model = ReferenceModel(dataset_path)
    try:
        ref_stats = ref_model.build_from_samples(exercise_id, camera="c17", fps=30, target_frames=50)
        print(f"  [OK] Reference built — {ref_stats['num_samples']} correct samples")
        print(f"  [OK] Shape: {ref_stats['mean_shape']}")
    except Exception as e:
        print(f"  [!] Reference model error: {e}")
        broker.stop()
        return {"error": str(e)}

    comparator = MovementComparator(ref_model.reference_mean, ref_model.reference_std)

    # ------------------------------------------------------------------
    # EDGE: Module 5 — Train Classifier
    # ------------------------------------------------------------------
    if train_classifier:
        _print_section("EDGE: Module 5 — Training Classifier (CNN + Threshold)")
        classifier = ExerciseClassifier(
            exercise_id=exercise_id,
            dataset_path=dataset_path,
            use_cnn=TF_AVAILABLE,
            use_threshold=True,
        )
        try:
            train_results = classifier.train(camera="c17", fps=30, train_split=0.8, epochs=30)
            print(f"  [OK] Trained on {train_results['total_samples']} samples")
            if 'threshold_classifier' in train_results:
                t = train_results['threshold_classifier']
                print(f"  [OK] Threshold: {t['threshold']:.4f}")
            if 'cnn_classifier' in train_results:
                c = train_results['cnn_classifier']
                print(f"  [OK] CNN accuracy: {c['train_accuracy']*100:.1f}%")
        except Exception as e:
            print(f"  [!] Classifier training error: {e}")
            classifier = None
    else:
        classifier = None

    # ------------------------------------------------------------------
    # DEMO: Process n_demo_samples through full enhanced pipeline
    # ------------------------------------------------------------------
    _print_section(f"EDGE + CLOUD: Processing {n_demo_samples} Demo Samples")

    import pandas as pd
    seg_path = os.path.join(dataset_path, "Segmentation.csv")
    seg_df = pd.read_csv(seg_path, sep=';')
    ex_num = int(exercise_id.replace("Ex", ""))
    ex_df = seg_df[seg_df['exercise_id'] == ex_num].head(n_demo_samples * 3)

    normaliser = SkeletonNormalizer()
    scorer = ConfidenceScorer()
    processed = 0
    session_ids_created = []
    sequences_for_lstm = []

    for _, row in ex_df.iterrows():
        if processed >= n_demo_samples:
            break

        # Load raw keypoints
        vid_id = row['video_id']
        fname = f"{vid_id}-c17-30fps.npy"
        fpath = os.path.join(dataset_path, "2d_joints", exercise_id, fname)
        if not os.path.exists(fpath):
            continue

        raw = np.load(fpath)
        seg = raw[int(row['first_frame']):int(row['last_frame'])]
        if len(seg) < 5:
            continue

        # Build keypoints (17, 3)
        kp = seg[:, :17, :2] if raw.shape[1] >= 17 else np.zeros((len(seg), 17, 2))
        conf = np.ones((len(kp), 17, 1))
        kp_with_conf = np.concatenate([kp, conf], axis=2)

        # --- Imputation (Module NEW) ---
        imputed_kp, imp_report = _imputer.impute_sequence(kp_with_conf)

        # --- Normalisation (Module 2) ---
        normalised = normaliser.normalize_sequence(imputed_kp)

        # --- Resample to 50 frames ---
        n_frames = len(normalised)
        indices = np.linspace(0, n_frames - 1, 50)
        resampled = np.zeros((50, 17, 2))
        for i, idx in enumerate(indices):
            lo = int(np.floor(idx))
            hi = min(lo + 1, n_frames - 1)
            w = idx - lo
            resampled[i] = (1 - w) * normalised[lo, :, :2] + w * normalised[hi, :, :2]

        sequences_for_lstm.append(resampled)

        # --- Module 4: Comparison ---
        comp_metrics = comparator.compare_sequence(resampled)
        rmse = comp_metrics['overall_rmse']

        # --- Module 5: Confidence + Classification ---
        confidence = scorer.score_from_rmse(rmse)
        if classifier is not None and classifier.is_trained:
            pred_result = classifier.predict(resampled)
            correctness = pred_result['final_prediction']
            method = pred_result['prediction_method']
        else:
            # Fallback threshold
            correctness = 1 if rmse < 0.30 else 0
            method = "Threshold-fallback"

        # --- Enhancement: Compensation Detection ---
        comp_det = _compensation.detect(resampled, exercise_id, ref_model.reference_mean)

        # --- Enhancement: Fluidity ---
        fluidity = _fluidity.analyze(resampled)

        # Build session data
        true_label = int(row['correctness'])
        session = {
            'patient_id': patient_id,
            'exercise_id': exercise_id,
            'video_id': vid_id,
            'repetition': int(row.get('repetition_number', 0)),
            'correctness': correctness,
            'true_label': true_label,
            'confidence': round(float(confidence), 4),
            'rmse': round(float(rmse), 4),
            'prediction_method': method,
            'compensation_found': comp_det['compensation_found'],
            'compensation_types': comp_det['types'],
            'compensation_severity': round(comp_det['severity'], 3),
            'fluidity_score': fluidity['overall_fluidity'],
            'fluidity_interpretation': fluidity['interpretation'],
            'fluidity_jerk': fluidity['jerk_score'],
            'imputation_total_missing': imp_report['total_missing'],
        }

        # --- PubSub: Edge publishes -> Cloud stores ---
        edge_pub.publish_session(session)
        broker.flush(timeout=1.0)

        # Track session IDs
        if db.total_sessions() > len(session_ids_created):
            all_s = db.get_all_sessions()
            if all_s:
                newest = sorted(all_s, key=lambda s: s.get('timestamp', ''), reverse=True)[0]
                if newest.get('session_id') not in session_ids_created:
                    session_ids_created.append(newest.get('session_id'))

        # Print result
        correct_icon = "OK" if correctness == true_label else "X"
        print(f"\n  [{correct_icon}] {vid_id} | True={'Correct' if true_label==1 else 'Incorrect'} "
              f"| Pred={'Correct' if correctness==1 else 'Incorrect'} | "
              f"RMSE={rmse:.3f} | Conf={confidence:.2f}")
        print(f"      Compensation: {comp_det['types'] if comp_det['compensation_found'] else 'None'}")
        print(f"      Fluidity:     {fluidity['overall_fluidity']:.2f} — {fluidity['interpretation']}")

        processed += 1

    # ------------------------------------------------------------------
    # Optional: Train LSTM fluidity on collected sequences
    # ------------------------------------------------------------------
    if TF_AVAILABLE and len(sequences_for_lstm) >= 10:
        _print_section("EDGE: LSTM Fluidity Training (proxy labels)")
        _fluidity.train_lstm(sequences_for_lstm, epochs=20)

    # ------------------------------------------------------------------
    # CLOUD: Population Benchmarking
    # ------------------------------------------------------------------
    _print_section("CLOUD: Population Benchmarking")

    broker.flush(timeout=2.0)
    pop_stats = bench.compute_population_stats(exercise_id)
    print(f"  [OK] Population sessions for {exercise_id}: {pop_stats.get('n', 0)}")
    if pop_stats.get('mean_rmse') is not None:
        print(f"  [OK] Population mean RMSE: {pop_stats['mean_rmse']:.4f}")
    if pop_stats.get('accuracy_rate') is not None:
        print(f"  [OK] Population accuracy:  {pop_stats['accuracy_rate']*100:.1f}%")

    patient_pct = bench.get_patient_percentile(patient_id, exercise_id, 'rmse')
    if patient_pct is not None:
        print(f"  [OK] {patient_id} RMSE percentile: {patient_pct:.0f}th (better than {patient_pct:.0f}% of population)")

    # ------------------------------------------------------------------
    # XML Export sample
    # ------------------------------------------------------------------
    _print_section("CLOUD: XML Export Sample")
    all_sessions = db.get_all_sessions()
    if all_sessions:
        last_session = sorted(all_sessions, key=lambda s: s.get('timestamp', ''), reverse=True)[0]
        xml_str = db.export_session_xml(last_session['session_id'])
        print(xml_str[:600])

    # ------------------------------------------------------------------
    broker.stop()
    print("\n" + "="*70)
    print("  END-TO-END PIPELINE COMPLETE")
    print(f"  Samples processed: {processed}")
    print(f"  Sessions in DB:    {db.total_sessions()}")
    print("="*70 + "\n")

    return {
        'exercise_id': exercise_id,
        'samples_processed': processed,
        'db_total_sessions': db.total_sessions(),
        'population_stats': pop_stats,
        'session_ids': session_ids_created,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Rehab Pipeline")
    parser.add_argument("--exercise", "-e", default="Ex1", help="Exercise ID (Ex1-Ex6)")
    parser.add_argument("--dataset", "-d", default="dataset", help="Dataset path")
    parser.add_argument("--patient", "-p", default="patient_demo", help="Patient ID")
    parser.add_argument("--db", default="rehab_data", help="Database directory")
    parser.add_argument("--samples", "-n", type=int, default=5, help="Samples to demo")
    parser.add_argument("--no-train", action="store_true", help="Skip classifier training")
    args = parser.parse_args()

    run_full_pipeline(
        exercise_id=args.exercise,
        dataset_path=args.dataset,
        patient_id=args.patient,
        db_path=args.db,
        n_demo_samples=args.samples,
        train_classifier=not args.no_train,
    )
