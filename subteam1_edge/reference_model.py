"""
Module 3: Reference Model Building
----------------------------------
Builds reference templates from correct exercise executions as per research paper.

Research Paper Reference:
    "A reference model is built from correct exercise executions.
    The mean joint positions across correct samples form the template.
    Observed movements are compared against this template."

Reference Model Components:
    1. Load correct execution samples (correctness = 1)
    2. Normalize all samples using Module 2
    3. Compute frame-wise mean positions (template)
    4. Compute standard deviation for variability bounds
"""

import os
import numpy as np
import pandas as pd
from normalization import SkeletonNormalizer


class ReferenceModel:
    """
    Reference model built from correct exercise executions.
    
    As per research paper:
    - Aggregate correct samples
    - Compute mean template per frame
    - Store statistics for comparison
    
    Attributes:
        reference_mean: Mean keypoint positions (frames, 17, 2)
        reference_std: Standard deviation (frames, 17, 2)
        num_samples: Number of correct samples used
    """
    
    def __init__(self, dataset_path="dataset"):
        """
        Initialize reference model.
        
        Parameters:
            dataset_path: Path to REHAB24-6 dataset root
        """
        self.dataset_path = dataset_path
        self.normalizer = SkeletonNormalizer()
        
        # Reference statistics
        self.reference_mean = None
        self.reference_std = None
        self.num_samples = 0
        self.target_frames = None
    
    def load_segmentation(self):
        """
        Load segmentation CSV with exercise metadata.
        
        Returns:
            DataFrame: Segmentation data with correctness labels
        """
        seg_path = os.path.join(self.dataset_path, "Segmentation.csv")
        
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        
        # Read CSV with semicolon separator
        df = pd.read_csv(seg_path, sep=';')
        return df
    
    def load_keypoints(self, exercise_id, video_id, camera="c17", fps=30):
        """
        Load keypoints for a specific video.
        
        Parameters:
            exercise_id: Exercise folder (Ex1, Ex2, etc.)
            video_id: Video identifier (PM_000, etc.)
            camera: Camera view (c17, c18)
            fps: Frame rate (30, 120)
            
        Returns:
            ndarray: Keypoints shape (frames, 17, 3)
        """
        filename = f"{video_id}-{camera}-{fps}fps.npy"
        filepath = os.path.join(self.dataset_path, "2d_joints", exercise_id, filename)
        
        if not os.path.exists(filepath):
            return None
        
        data = np.load(filepath)
        
        # Convert to 17 keypoints with confidence
        # Dataset has 26 joints, use first 17
        if data.shape[1] >= 17:
            keypoints = data[:, :17, :2]
        else:
            keypoints = np.zeros((data.shape[0], 17, 2))
            keypoints[:, :data.shape[1], :] = data[:, :, :2]
        
        # Add confidence = 1.0 (ground truth)
        confidence = np.ones((keypoints.shape[0], 17, 1))
        keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)
        
        return keypoints_with_conf
    
    def extract_repetition(self, keypoints, first_frame, last_frame):
        """
        Extract frames for a single repetition.
        
        Parameters:
            keypoints: Full video keypoints (frames, 17, 3)
            first_frame: Start frame index
            last_frame: End frame index
            
        Returns:
            ndarray: Repetition keypoints
        """
        # Ensure valid frame range
        first_frame = max(0, first_frame)
        last_frame = min(len(keypoints), last_frame)
        
        return keypoints[first_frame:last_frame]
    
    def resample_to_length(self, keypoints, target_length):
        """
        Resample keypoint sequence to fixed length using interpolation.
        
        This allows comparing repetitions of different durations.
        
        Parameters:
            keypoints: Shape (frames, 17, 3)
            target_length: Desired number of frames
            
        Returns:
            ndarray: Resampled to (target_length, 17, 3)
        """
        current_length = len(keypoints)
        
        if current_length == target_length:
            return keypoints
        
        # Create interpolation indices
        indices = np.linspace(0, current_length - 1, target_length)
        
        resampled = np.zeros((target_length, 17, 3))
        
        for i in range(target_length):
            idx = indices[i]
            lower = int(np.floor(idx))
            upper = min(lower + 1, current_length - 1)
            weight = idx - lower
            
            # Linear interpolation
            resampled[i] = (1 - weight) * keypoints[lower] + weight * keypoints[upper]
        
        return resampled
    
    def build_from_samples(self, exercise_id, camera="c17", fps=30, target_frames=50):
        """
        Build reference model from correct samples of an exercise.
        
        Parameters:
            exercise_id: Exercise to build reference for (Ex1, Ex2, etc.)
            camera: Camera view to use
            fps: Frame rate version
            target_frames: Standard length for alignment
            
        Returns:
            dict: Training statistics
        """
        # Convert exercise_id format
        if isinstance(exercise_id, str) and exercise_id.startswith("Ex"):
            exercise_num = int(exercise_id[2:])
        else:
            exercise_num = int(exercise_id)
            exercise_id = f"Ex{exercise_num}"
        
        self.target_frames = target_frames
        
        # Load segmentation
        seg_df = self.load_segmentation()
        
        # Filter for this exercise and correct samples only
        correct_samples = seg_df[
            (seg_df['exercise_id'] == exercise_num) & 
            (seg_df['correctness'] == 1)
        ].copy()
        
        if len(correct_samples) == 0:
            raise ValueError(f"No correct samples found for {exercise_id}")
        
        print(f"Building reference model for {exercise_id}")
        print(f"Found {len(correct_samples)} correct repetitions")
        
        # Collect all normalized samples
        all_samples = []
        
        for _, row in correct_samples.iterrows():
            video_id = row['video_id']
            first_frame = int(row['first_frame'])
            last_frame = int(row['last_frame'])
            
            # Load video keypoints
            keypoints = self.load_keypoints(exercise_id, video_id, camera, fps)
            
            if keypoints is None:
                continue
            
            # Extract repetition
            repetition = self.extract_repetition(keypoints, first_frame, last_frame)
            
            if len(repetition) < 5:  # Skip very short segments
                continue
            
            # Normalize
            normalized = self.normalizer.normalize_sequence(repetition)
            
            # Resample to fixed length
            resampled = self.resample_to_length(normalized, target_frames)
            
            all_samples.append(resampled)
        
        if len(all_samples) == 0:
            raise ValueError(f"Could not load any samples for {exercise_id}")
        
        # Stack and compute statistics
        all_samples = np.array(all_samples)  # (num_samples, frames, 17, 3)
        
        self.num_samples = len(all_samples)
        
        # Compute mean and std (only for x, y coordinates)
        self.reference_mean = np.mean(all_samples[:, :, :, :2], axis=0)  # (frames, 17, 2)
        self.reference_std = np.std(all_samples[:, :, :, :2], axis=0)    # (frames, 17, 2)
        
        # Avoid division by zero in std
        self.reference_std = np.maximum(self.reference_std, 1e-6)
        
        print(f"Reference model built from {self.num_samples} samples")
        print(f"Reference shape: {self.reference_mean.shape}")
        
        return {
            'exercise_id': exercise_id,
            'num_samples': self.num_samples,
            'target_frames': target_frames,
            'mean_shape': self.reference_mean.shape,
            'std_shape': self.reference_std.shape
        }
    
    def get_reference_frame(self, frame_idx):
        """
        Get reference keypoints for a specific frame.
        
        Parameters:
            frame_idx: Frame index (0 to target_frames-1)
            
        Returns:
            tuple: (mean_keypoints, std_keypoints)
        """
        if self.reference_mean is None:
            raise ValueError("Reference model not built. Call build_from_samples first.")
        
        return self.reference_mean[frame_idx], self.reference_std[frame_idx]
    
    def save(self, filepath):
        """Save reference model to file."""
        np.savez(
            filepath,
            reference_mean=self.reference_mean,
            reference_std=self.reference_std,
            num_samples=self.num_samples,
            target_frames=self.target_frames
        )
    
    def load(self, filepath):
        """Load reference model from file."""
        data = np.load(filepath)
        self.reference_mean = data['reference_mean']
        self.reference_std = data['reference_std']
        self.num_samples = int(data['num_samples'])
        self.target_frames = int(data['target_frames'])


if __name__ == "__main__":
    # Test reference model building
    print("Testing Reference Model Module...")
    
    dataset_path = "dataset"
    
    if os.path.exists(dataset_path):
        model = ReferenceModel(dataset_path)
        
        try:
            stats = model.build_from_samples("Ex1", camera="c17", fps=30)
            print(f"\nTraining Stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Dataset not found at: {dataset_path}")
