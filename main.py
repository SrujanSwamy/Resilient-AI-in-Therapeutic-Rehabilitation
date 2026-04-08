"""
Main Pipeline - Rehabilitation Exercise Analysis
-------------------------------------------------
Integration of all 5 modules as per research paper:

Module 1: MoveNet Pose Estimation
Module 2: Skeleton Normalization  
Module 3: Reference Model Building
Module 4: Movement Comparison (Euclidean, MSE, RMSE)
Module 5: Confidence Scoring + Classification (CNN + Threshold)

Research Paper: "Resilient AI in Therapeutic Rehabilitation"

Usage:
    python main.py --exercise Ex1                    # Train and demo
    python main.py --exercise Ex1 --evaluate         # Full evaluation
    python main.py --exercise Ex1 --train-only       # Train without demo
"""

import os
import sys
import argparse
import numpy as np

# Add subteam1_edge to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'subteam1_edge'))

from movenet import MoveNetPoseEstimator, KEYPOINT_NAMES, TF_AVAILABLE
from normalization import SkeletonNormalizer
from reference_model import ReferenceModel
from comparison import MovementComparator, compute_rmse
from confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD
from classifier import ExerciseClassifier, CNNClassifier, ThresholdClassifier


def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print("  REHABILITATION EXERCISE ANALYSIS SYSTEM")
    print("  Based on: 'Resilient AI in Therapeutic Rehabilitation'")
    print("="*70)
    print(f"  TensorFlow Available: {TF_AVAILABLE}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  MoveNet Keypoints: {len(KEYPOINT_NAMES)}")
    print("="*70 + "\n")


def demo_module1():
    """Demonstrate Module 1: MoveNet Pose Estimation."""
    print("\n" + "-"*60)
    print("MODULE 1: MoveNet Pose Estimation")
    print("-"*60)
    
    if not TF_AVAILABLE:
        print("  [!] TensorFlow not available. Using dataset keypoints.")
        print("  [!] Install with: pip install tensorflow tensorflow-hub")
        return
    
    print("  Loading MoveNet Thunder model...")
    estimator = MoveNetPoseEstimator()
    
    if estimator.model is not None:
        print("  [✓] MoveNet Thunder loaded successfully")
        print(f"  [✓] Input size: {estimator.input_size}x{estimator.input_size}")
        print(f"  [✓] Output: 17 keypoints (x, y, confidence)")
    else:
        print("  [!] MoveNet loading failed")


def demo_module2():
    """Demonstrate Module 2: Skeleton Normalization."""
    print("\n" + "-"*60)
    print("MODULE 2: Skeleton Normalization")
    print("-"*60)
    
    # Create sample keypoints
    np.random.seed(42)
    sample = np.random.rand(17, 3)
    sample[:, 2] = 1.0  # confidence
    
    normalizer = SkeletonNormalizer()
    normalized = normalizer.normalize_frame(sample)
    
    print(f"  Original hip center: [{normalizer.hip_center[0]:.4f}, {normalizer.hip_center[1]:.4f}]")
    print(f"  Torso length: {normalizer.torso_length:.4f}")
    
    # Verify normalization
    new_hip = (normalized[11, :2] + normalized[12, :2]) / 2
    print(f"  Normalized hip center: [{new_hip[0]:.4f}, {new_hip[1]:.4f}] (should be ~0)")
    print("  [✓] Normalization complete")


def demo_module3(dataset_path: str, exercise_id: str):
    """Demonstrate Module 3: Reference Model Building."""
    print("\n" + "-"*60)
    print("MODULE 3: Reference Model Building")
    print("-"*60)
    
    model = ReferenceModel(dataset_path)
    
    try:
        stats = model.build_from_samples(exercise_id, camera="c17", fps=30, target_frames=50)
        print(f"  [✓] Reference model built for {exercise_id}")
        print(f"  [✓] Samples used: {stats['num_samples']}")
        print(f"  [✓] Target frames: {stats['target_frames']}")
        print(f"  [✓] Reference shape: {stats['mean_shape']}")
        return model
    except Exception as e:
        print(f"  [!] Error: {e}")
        return None


def demo_module4(reference_model: ReferenceModel, dataset_path: str, exercise_id: str):
    """Demonstrate Module 4: Movement Comparison."""
    print("\n" + "-"*60)
    print("MODULE 4: Movement Comparison")
    print("-"*60)
    
    if reference_model is None:
        print("  [!] Reference model not available")
        return
    
    # Create comparator
    comparator = MovementComparator(
        reference_model.reference_mean,
        reference_model.reference_std
    )
    
    # Load a sample to compare
    import pandas as pd
    seg_path = os.path.join(dataset_path, "Segmentation.csv")
    seg_df = pd.read_csv(seg_path, sep=';')
    
    if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
        exercise_num = int(exercise_id[2:])
    else:
        exercise_num = int(exercise_id)
    
    # Get one correct and one incorrect sample
    correct_sample = seg_df[(seg_df['exercise_id'] == exercise_num) & (seg_df['correctness'] == 1)].iloc[0]
    incorrect_sample = seg_df[(seg_df['exercise_id'] == exercise_num) & (seg_df['correctness'] == 0)].iloc[0]
    
    normalizer = SkeletonNormalizer()
    
    def load_and_process(row):
        filename = f"{row['video_id']}-c17-30fps.npy"
        filepath = os.path.join(dataset_path, "2d_joints", exercise_id, filename)
        if not os.path.exists(filepath):
            return None
        data = np.load(filepath)
        segment = data[int(row['first_frame']):int(row['last_frame'])]
        if len(segment) < 5:
            return None
        # Convert to 17 keypoints
        kp = segment[:, :17, :2] if segment.shape[1] >= 17 else np.zeros((len(segment), 17, 2))
        conf = np.ones((len(kp), 17, 1))
        kp_with_conf = np.concatenate([kp, conf], axis=2)
        normalized = normalizer.normalize_sequence(kp_with_conf)
        # Resample
        target = 50
        indices = np.linspace(0, len(normalized)-1, target)
        resampled = np.zeros((target, 17, 2))
        for i in range(target):
            idx = indices[i]
            lower = int(np.floor(idx))
            upper = min(lower + 1, len(normalized) - 1)
            weight = idx - lower
            resampled[i] = (1 - weight) * normalized[lower, :, :2] + weight * normalized[upper, :, :2]
        return resampled
    
    correct_kp = load_and_process(correct_sample)
    incorrect_kp = load_and_process(incorrect_sample)
    
    if correct_kp is not None:
        metrics_correct = comparator.compare_sequence(correct_kp)
        print(f"  Correct Sample ({correct_sample['video_id']}):")
        print(f"    RMSE: {metrics_correct['overall_rmse']:.4f}")
        print(f"    Mean Euclidean: {metrics_correct['mean_euclidean']:.4f}")
    
    if incorrect_kp is not None:
        metrics_incorrect = comparator.compare_sequence(incorrect_kp)
        print(f"  Incorrect Sample ({incorrect_sample['video_id']}):")
        print(f"    RMSE: {metrics_incorrect['overall_rmse']:.4f}")
        print(f"    Mean Euclidean: {metrics_incorrect['mean_euclidean']:.4f}")
    
    print("  [✓] Comparison metrics computed")


def demo_module5(dataset_path: str, exercise_id: str, evaluate: bool = False):
    """Demonstrate Module 5: Confidence + Classification."""
    print("\n" + "-"*60)
    print("MODULE 5: Confidence Scoring & Classification")
    print("-"*60)
    
    # Confidence scoring demo
    scorer = ConfidenceScorer()
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    test_rmse = [0.02, 0.05, 0.1, 0.2]
    print("\n  RMSE → Confidence mapping:")
    for rmse in test_rmse:
        conf = scorer.score_from_rmse(rmse)
        status = "Reliable" if conf >= CONFIDENCE_THRESHOLD else "Warning"
        print(f"    RMSE={rmse:.2f} → Confidence={conf:.3f} → {status}")
    
    # Full classifier
    print("\n  Training classifiers (CNN + Threshold)...")
    
    classifier = ExerciseClassifier(
        exercise_id=exercise_id,
        dataset_path=dataset_path,
        use_cnn=TF_AVAILABLE,
        use_threshold=True
    )
    
    try:
        train_results = classifier.train(
            camera="c17", 
            fps=30, 
            train_split=0.8,
            epochs=50  # Reduced for demo (paper uses 200)
        )
        
        print(f"\n  [✓] Training Complete")
        print(f"      Total samples: {train_results['total_samples']}")
        
        if 'threshold_classifier' in train_results:
            t = train_results['threshold_classifier']
            print(f"      Threshold: {t['threshold']:.4f}")
        
        if 'cnn_classifier' in train_results:
            c = train_results['cnn_classifier']
            print(f"      CNN Train Acc: {c['train_accuracy']*100:.1f}%")
            if c['val_accuracy']:
                print(f"      CNN Val Acc: {c['val_accuracy']*100:.1f}%")
        
        # Demo prediction
        print("\n  Demo Predictions:")
        import pandas as pd
        seg_df = pd.read_csv(os.path.join(dataset_path, "Segmentation.csv"), sep=';')
        
        if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
            exercise_num = int(exercise_id[2:])
        else:
            exercise_num = int(exercise_id)
        
        samples = seg_df[seg_df['exercise_id'] == exercise_num].head(5)
        
        for _, row in samples.iterrows():
            sample = classifier.load_sample(
                row['video_id'],
                int(row['first_frame']),
                int(row['last_frame'])
            )
            
            if sample is None:
                continue
            
            pred = classifier.predict(sample)
            true_label = "Correct" if row['correctness'] == 1 else "Incorrect"
            
            print(f"    {row['video_id']}: True={true_label}, "
                  f"Pred={pred['label']}, "
                  f"Conf={pred['confidence']:.2f}, "
                  f"Method={pred['prediction_method']}")
        
        # Full evaluation if requested
        if evaluate:
            print("\n" + "-"*60)
            print("FULL EVALUATION")
            print("-"*60)
            eval_results = classifier.evaluate(camera="c17", fps=30)
        
    except Exception as e:
        print(f"  [!] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rehabilitation Exercise Analysis System"
    )
    parser.add_argument(
        '--exercise', '-e',
        type=str,
        default='Ex1',
        help='Exercise ID (Ex1-Ex6)'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='dataset',
        help='Path to REHAB24-6 dataset'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run full evaluation after training'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train, skip demonstrations'
    )
    parser.add_argument(
        '--module',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Demo specific module only'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at '{args.dataset}'")
        print("Please ensure the REHAB24-6 dataset is in the correct location.")
        sys.exit(1)
    
    print_header()
    
    # Run specific module or all
    if args.module:
        if args.module == 1:
            demo_module1()
        elif args.module == 2:
            demo_module2()
        elif args.module == 3:
            demo_module3(args.dataset, args.exercise)
        elif args.module == 4:
            ref_model = demo_module3(args.dataset, args.exercise)
            demo_module4(ref_model, args.dataset, args.exercise)
        elif args.module == 5:
            demo_module5(args.dataset, args.exercise, args.evaluate)
    elif args.train_only:
        # Just train Module 5
        demo_module5(args.dataset, args.exercise, args.evaluate)
    else:
        # Run all modules in sequence
        print("Running complete pipeline (Modules 1-5)...")
        
        demo_module1()
        demo_module2()
        ref_model = demo_module3(args.dataset, args.exercise)
        demo_module4(ref_model, args.dataset, args.exercise)
        demo_module5(args.dataset, args.exercise, args.evaluate)
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
