"""
Demo: Classification Output for All Exercises
----------------------------------------------
Shows classification results for samples across all 6 exercises.
Demonstrates the complete pipeline: Normalization → Comparison → Classification

Usage:
    python demo_classify.py
    python demo_classify.py --exercise Ex1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'subteam1_edge'))

from normalization import SkeletonNormalizer
from comparison import compute_rmse
from confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD


# Paths
DATASET_PATH = "dataset"


def load_segmentation():
    """Load segmentation CSV."""
    seg_path = os.path.join(DATASET_PATH, "Segmentation.csv")
    return pd.read_csv(seg_path, sep=';')


def load_sample(exercise_id, video_id, first_frame, last_frame, camera="c17", fps=30):
    """Load and preprocess a single sample."""
    normalizer = SkeletonNormalizer()
    
    filename = f"{video_id}-{camera}-{fps}fps.npy"
    filepath = os.path.join(DATASET_PATH, "2d_joints", exercise_id, filename)
    
    if not os.path.exists(filepath):
        return None
    
    data = np.load(filepath)
    
    # Extract segment
    first_frame = max(0, first_frame)
    last_frame = min(len(data), last_frame)
    segment = data[first_frame:last_frame]
    
    if len(segment) < 5:
        return None
    
    # Convert to 17 keypoints
    if segment.shape[1] >= 17:
        keypoints = segment[:, :17, :2]
    else:
        keypoints = np.zeros((len(segment), 17, 2))
        keypoints[:, :segment.shape[1], :] = segment[:, :, :2]
    
    # Add confidence
    confidence = np.ones((len(keypoints), 17, 1))
    keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)
    
    # Normalize
    normalized = normalizer.normalize_sequence(keypoints_with_conf)
    
    # Resample to 50 frames
    target_frames = 50
    current_length = len(normalized)
    indices = np.linspace(0, current_length - 1, target_frames)
    
    resampled = np.zeros((target_frames, 17, 2))
    for i in range(target_frames):
        idx = indices[i]
        lower = int(np.floor(idx))
        upper = min(lower + 1, current_length - 1)
        weight = idx - lower
        resampled[i] = (1 - weight) * normalized[lower, :, :2] + weight * normalized[upper, :, :2]
    
    return resampled


def build_reference(exercise_id, seg_df, camera="c17", fps=30):
    """Build reference model from correct samples."""
    if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
        exercise_num = int(exercise_id[2:])
    else:
        exercise_num = int(exercise_id)
        exercise_id = f"Ex{exercise_num}"
    
    correct_samples = seg_df[
        (seg_df['exercise_id'] == exercise_num) & 
        (seg_df['correctness'] == 1)
    ]
    
    all_samples = []
    
    for _, row in correct_samples.iterrows():
        sample = load_sample(
            exercise_id,
            row['video_id'],
            int(row['first_frame']),
            int(row['last_frame']),
            camera, fps
        )
        if sample is not None:
            all_samples.append(sample)
    
    if len(all_samples) == 0:
        return None, None
    
    all_samples = np.array(all_samples)
    reference_mean = np.mean(all_samples, axis=0)
    reference_std = np.std(all_samples, axis=0)
    
    return reference_mean, reference_std


def demo_exercise(exercise_id, seg_df, num_samples=10):
    """Demo classification for one exercise."""
    
    print(f"\n{'='*70}")
    print(f"EXERCISE: {exercise_id}")
    print('='*70)
    
    if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
        exercise_num = int(exercise_id[2:])
    else:
        exercise_num = int(exercise_id)
        exercise_id = f"Ex{exercise_num}"
    
    # Build reference
    print("\n[1] Building Reference Model from correct samples...")
    reference_mean, reference_std = build_reference(exercise_id, seg_df)
    
    if reference_mean is None:
        print("    ERROR: Could not build reference model")
        return None
    
    print(f"    Reference shape: {reference_mean.shape}")
    
    # Calculate threshold from correct samples
    correct_samples = seg_df[
        (seg_df['exercise_id'] == exercise_num) & 
        (seg_df['correctness'] == 1)
    ]
    incorrect_samples = seg_df[
        (seg_df['exercise_id'] == exercise_num) & 
        (seg_df['correctness'] == 0)
    ]
    
    correct_rmses = []
    for _, row in correct_samples.head(20).iterrows():
        sample = load_sample(exercise_id, row['video_id'], 
                            int(row['first_frame']), int(row['last_frame']))
        if sample is not None:
            rmse = compute_rmse(reference_mean, sample)
            correct_rmses.append(rmse)
    
    incorrect_rmses = []
    for _, row in incorrect_samples.head(20).iterrows():
        sample = load_sample(exercise_id, row['video_id'],
                            int(row['first_frame']), int(row['last_frame']))
        if sample is not None:
            rmse = compute_rmse(reference_mean, sample)
            incorrect_rmses.append(rmse)
    
    if len(correct_rmses) == 0 or len(incorrect_rmses) == 0:
        print("    ERROR: Not enough samples for threshold calculation")
        return None
    
    # Threshold at midpoint
    threshold = (np.mean(correct_rmses) + np.mean(incorrect_rmses)) / 2
    
    print(f"    Correct RMSE mean: {np.mean(correct_rmses):.4f}")
    print(f"    Incorrect RMSE mean: {np.mean(incorrect_rmses):.4f}")
    print(f"    Classification Threshold: {threshold:.4f}")
    
    # Classify samples
    print(f"\n[2] Classification Results...")
    print("-"*70)
    print(f"{'Video':<12} {'Rep':<5} {'Actual':<12} {'Predicted':<12} {'RMSE':<10} {'Conf':<8} {'Match'}")
    print("-"*70)
    
    # Get mixed samples
    exercise_df = seg_df[seg_df['exercise_id'] == exercise_num].head(num_samples)
    
    confidence_scorer = ConfidenceScorer()
    
    correct_predictions = 0
    total = 0
    
    for _, row in exercise_df.iterrows():
        sample = load_sample(
            exercise_id,
            row['video_id'],
            int(row['first_frame']),
            int(row['last_frame'])
        )
        
        if sample is None:
            continue
        
        # Calculate RMSE
        rmse = compute_rmse(reference_mean, sample)
        
        # Classify
        predicted = 1 if rmse < threshold else 0
        predicted_label = "Correct" if predicted == 1 else "Incorrect"
        
        # Actual
        actual = int(row['correctness'])
        actual_label = "Correct" if actual == 1 else "Incorrect"
        
        # Confidence
        confidence = confidence_scorer.score_from_rmse(rmse)
        
        # Match
        match = "✓" if predicted == actual else "✗"
        if predicted == actual:
            correct_predictions += 1
        total += 1
        
        print(f"{row['video_id']:<12} {row['repetition_number']:<5} {actual_label:<12} {predicted_label:<12} {rmse:<10.4f} {confidence:<8.2%} {match}")
    
    # Summary
    if total > 0:
        accuracy = correct_predictions / total
        print("-"*70)
        print(f"Accuracy: {accuracy:.1%} ({correct_predictions}/{total})")
    
    return accuracy if total > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Classification Demo")
    parser.add_argument('--exercise', '-e', type=str, default=None,
                       help='Single exercise (Ex1-Ex6) or all if not specified')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of samples per exercise')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  REHABILITATION EXERCISE CLASSIFICATION DEMO")
    print("  Research Paper: Resilient AI in Therapeutic Rehabilitation")
    print("="*70)
    
    # Load segmentation
    seg_df = load_segmentation()
    
    results = {}
    
    if args.exercise:
        # Single exercise
        exercises = [args.exercise]
    else:
        # All exercises
        exercises = ['Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6']
    
    for exercise in exercises:
        try:
            accuracy = demo_exercise(exercise, seg_df, args.samples)
            if accuracy is not None:
                results[exercise] = accuracy
        except Exception as e:
            print(f"Error processing {exercise}: {e}")
    
    # Overall summary
    if len(results) > 1:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        for ex, acc in results.items():
            print(f"  {ex}: {acc:.1%}")
        print(f"\n  Average Accuracy: {np.mean(list(results.values())):.1%}")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
