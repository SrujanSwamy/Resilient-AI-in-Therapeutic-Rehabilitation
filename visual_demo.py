"""
Visual Demo: Skeleton Overlay with Classification
--------------------------------------------------
Displays video with skeleton overlay.
- GREEN skeleton = Correct posture
- BLUE skeleton = Incorrect posture

Usage:
    python visual_demo.py
    python visual_demo.py --exercise Ex1 --video PM_000
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'subteam1_edge'))

from normalization import SkeletonNormalizer
from comparison import compute_rmse

# Paths
DATASET_PATH = "dataset"

# MoveNet skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


def load_segmentation():
    """Load segmentation CSV."""
    seg_path = os.path.join(DATASET_PATH, "Segmentation.csv")
    return pd.read_csv(seg_path, sep=';')


def load_joints_data(exercise_id, video_id, camera="c17", fps=30):
    """Load full joint data for a video."""
    filename = f"{video_id}-{camera}-{fps}fps.npy"
    filepath = os.path.join(DATASET_PATH, "2d_joints", exercise_id, filename)
    
    if not os.path.exists(filepath):
        return None
    
    data = np.load(filepath)
    return data


def load_video(exercise_id, video_id, camera="c17", fps=30):
    """Load video file."""
    # Videos use "Camera17" format, not "c17"
    camera_name = "Camera17" if camera == "c17" else "Camera18"
    video_name = f"{video_id}-{camera_name}-{fps}fps.mp4"
    video_path = os.path.join(DATASET_PATH, "videos", exercise_id, video_name)
    
    if not os.path.exists(video_path):
        # Try other extensions
        for ext in ['.avi', '.mov', '.mkv']:
            alt_path = video_path.replace('.mp4', ext)
            if os.path.exists(alt_path):
                video_path = alt_path
                break
        else:
            return None
    
    return cv2.VideoCapture(video_path)


def build_reference(exercise_id, seg_df, camera="c17", fps=30):
    """Build reference model from correct samples."""
    normalizer = SkeletonNormalizer()
    
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
    
    for _, row in correct_samples.head(10).iterrows():
        filename = f"{row['video_id']}-{camera}-{fps}fps.npy"
        filepath = os.path.join(DATASET_PATH, "2d_joints", exercise_id, filename)
        
        if not os.path.exists(filepath):
            continue
        
        data = np.load(filepath)
        
        first_frame = max(0, int(row['first_frame']))
        last_frame = min(len(data), int(row['last_frame']))
        segment = data[first_frame:last_frame]
        
        if len(segment) < 5:
            continue
        
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
        
        all_samples.append(resampled)
    
    if len(all_samples) == 0:
        return None
    
    all_samples = np.array(all_samples)
    reference_mean = np.mean(all_samples, axis=0)
    
    return reference_mean


def calculate_threshold(exercise_id, seg_df, reference_mean):
    """Calculate classification threshold."""
    normalizer = SkeletonNormalizer()
    
    if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
        exercise_num = int(exercise_id[2:])
    else:
        exercise_num = int(exercise_id)
        exercise_id = f"Ex{exercise_num}"
    
    correct_samples = seg_df[
        (seg_df['exercise_id'] == exercise_num) & 
        (seg_df['correctness'] == 1)
    ]
    incorrect_samples = seg_df[
        (seg_df['exercise_id'] == exercise_num) & 
        (seg_df['correctness'] == 0)
    ]
    
    def get_sample_rmse(row):
        filename = f"{row['video_id']}-c17-30fps.npy"
        filepath = os.path.join(DATASET_PATH, "2d_joints", exercise_id, filename)
        
        if not os.path.exists(filepath):
            return None
        
        data = np.load(filepath)
        
        first_frame = max(0, int(row['first_frame']))
        last_frame = min(len(data), int(row['last_frame']))
        segment = data[first_frame:last_frame]
        
        if len(segment) < 5:
            return None
        
        # Process
        if segment.shape[1] >= 17:
            keypoints = segment[:, :17, :2]
        else:
            keypoints = np.zeros((len(segment), 17, 2))
            keypoints[:, :segment.shape[1], :] = segment[:, :, :2]
        
        confidence = np.ones((len(keypoints), 17, 1))
        keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)
        normalized = normalizer.normalize_sequence(keypoints_with_conf)
        
        # Resample
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
        
        return compute_rmse(reference_mean, resampled)
    
    correct_rmses = [r for r in [get_sample_rmse(row) for _, row in correct_samples.head(10).iterrows()] if r is not None]
    incorrect_rmses = [r for r in [get_sample_rmse(row) for _, row in incorrect_samples.head(10).iterrows()] if r is not None]
    
    if len(correct_rmses) == 0 or len(incorrect_rmses) == 0:
        return 0.15  # Default
    
    return (np.mean(correct_rmses) + np.mean(incorrect_rmses)) / 2


def draw_skeleton(frame, keypoints, color, thickness=2):
    """Draw skeleton on frame."""
    h, w = frame.shape[:2]
    
    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
            pt2 = (int(keypoints[j, 0] * w), int(keypoints[j, 1] * h))
            
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for i in range(min(17, len(keypoints))):
        pt = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, pt, 4, color, -1)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1)
    
    return frame


def visualize_sample(exercise_id, video_id, first_frame, last_frame, is_correct, 
                    reference_mean, threshold, joints_data):
    """Visualize a single sample with classification overlay."""
    
    normalizer = SkeletonNormalizer()
    
    # Colors: GREEN for correct, RED/BLUE for incorrect  
    COLOR_CORRECT = (0, 255, 0)    # Green
    COLOR_INCORRECT = (255, 0, 0)  # Blue (BGR)
    
    # Load video
    cap = load_video(exercise_id, video_id)
    
    if cap is None:
        print(f"Could not load video for {video_id}")
        # Just use joints data on blank frame
        display_joints_only = True
        width, height = 640, 480
        fps = 30
    else:
        display_joints_only = False
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Process segment
    segment = joints_data[first_frame:last_frame]
    if segment.shape[1] >= 17:
        keypoints = segment[:, :17, :2]
    else:
        keypoints = np.zeros((len(segment), 17, 2))
        keypoints[:, :segment.shape[1], :] = segment[:, :, :2]
    
    # Add confidence and normalize
    confidence = np.ones((len(keypoints), 17, 1))
    keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)
    normalized = normalizer.normalize_sequence(keypoints_with_conf)
    
    # Resample for RMSE calculation
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
    
    # Calculate RMSE and classify
    rmse = compute_rmse(reference_mean, resampled)
    predicted_correct = rmse < threshold
    
    # Determine color
    color = COLOR_CORRECT if predicted_correct else COLOR_INCORRECT
    
    # Window name
    window_name = f"{exercise_id} - {video_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print(f"\n  Video: {video_id}")
    print(f"  Actual: {'Correct' if is_correct else 'Incorrect'}")
    print(f"  Predicted: {'Correct' if predicted_correct else 'Incorrect'}")
    print(f"  RMSE: {rmse:.4f}, Threshold: {threshold:.4f}")
    print(f"  Color: {'GREEN' if predicted_correct else 'BLUE'}")
    print(f"\n  Press 'Q' to skip, 'ESC' to quit all")
    
    # Playback
    frame_idx = 0
    
    while frame_idx < len(segment):
        if display_joints_only:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
        
        # Draw skeleton
        kp = keypoints[frame_idx]
        frame = draw_skeleton(frame, kp, color, thickness=3)
        
        # Add text overlay
        status_text = "CORRECT" if predicted_correct else "INCORRECT"
        cv2.putText(frame, f"{exercise_id}: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"RMSE: {rmse:.4f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{len(segment)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Ground truth
        cv2.putText(frame, f"Actual: {'Correct' if is_correct else 'Incorrect'}", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            if not display_joints_only:
                cap.release()
            return False  # Signal to quit all
        
        frame_idx += 1
    
    if not display_joints_only:
        cap.release()
    cv2.destroyWindow(window_name)
    
    return True  # Continue


def main():
    parser = argparse.ArgumentParser(description="Visual Classification Demo")
    parser.add_argument('--exercise', '-e', type=str, default='Ex1',
                       help='Exercise ID (Ex1-Ex6)')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Specific video ID (e.g., PM_000)')
    parser.add_argument('--samples', '-n', type=int, default=5,
                       help='Number of samples to show')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  VISUAL CLASSIFICATION DEMO")
    print("  GREEN skeleton = Correct posture")
    print("  BLUE skeleton = Incorrect posture")
    print("="*60)
    
    # Load segmentation
    seg_df = load_segmentation()
    
    exercise_id = args.exercise
    if not exercise_id.startswith("Ex"):
        exercise_id = f"Ex{exercise_id}"
    
    exercise_num = int(exercise_id[2:])
    
    print(f"\n[1] Building reference model for {exercise_id}...")
    reference_mean = build_reference(exercise_id, seg_df)
    
    if reference_mean is None:
        print("ERROR: Could not build reference model")
        return
    
    print(f"    Done! Shape: {reference_mean.shape}")
    
    print("\n[2] Calculating classification threshold...")
    threshold = calculate_threshold(exercise_id, seg_df, reference_mean)
    print(f"    Threshold: {threshold:.4f}")
    
    # Get samples to visualize
    exercise_df = seg_df[seg_df['exercise_id'] == exercise_num]
    
    if args.video:
        # Filter to specific video
        exercise_df = exercise_df[exercise_df['video_id'] == args.video]
    
    exercise_df = exercise_df.head(args.samples)
    
    print(f"\n[3] Visualizing {len(exercise_df)} samples...")
    print("    (Press 'Q' to skip sample, 'ESC' to quit)")
    
    for _, row in exercise_df.iterrows():
        video_id = row['video_id']
        first_frame = int(row['first_frame'])
        last_frame = int(row['last_frame'])
        is_correct = int(row['correctness']) == 1
        
        # Load joints
        joints_data = load_joints_data(exercise_id, video_id)
        if joints_data is None:
            print(f"    Skipping {video_id} - no joint data")
            continue
        
        # Visualize
        cont = visualize_sample(
            exercise_id, video_id, first_frame, last_frame,
            is_correct, reference_mean, threshold, joints_data
        )
        
        if not cont:
            break
    
    cv2.destroyAllWindows()
    print("\n  Demo complete!")


if __name__ == "__main__":
    main()
