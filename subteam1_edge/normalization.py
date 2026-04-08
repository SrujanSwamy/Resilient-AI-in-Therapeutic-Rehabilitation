"""
Module 2: Skeleton Normalization
--------------------------------
Skeleton normalization for scale and position invariance as per research paper.

Research Paper Reference:
    "Skeletal positions are normalized to the hip center and scaled 
    by torso length to enable subject-invariant comparison."

Normalization Steps (as per paper):
    1. Calculate hip center (midpoint of left_hip and right_hip)
    2. Translate all keypoints so hip center is at origin (0, 0)
    3. Calculate torso length (hip center to shoulder center distance)
    4. Scale all coordinates by torso length

This allows comparison between subjects of different body sizes.
"""

import numpy as np


# MoveNet 17 keypoint indices (as per paper)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class SkeletonNormalizer:
    """
    Normalizes 17-keypoint skeleton for scale and position invariance.
    
    As per research paper:
    - Center skeleton at hip center (0, 0)
    - Scale by torso length for size invariance
    
    Attributes:
        hip_center: Computed center point between hips
        torso_length: Distance from hip center to shoulder center
    """
    
    def __init__(self):
        """Initialize normalizer."""
        self.hip_center = None
        self.torso_length = None
    
    def compute_hip_center(self, keypoints):
        """
        Compute hip center as midpoint of left and right hip.
        
        Parameters:
            keypoints: Shape (17, 2) or (17, 3) with (x, y) or (x, y, conf)
            
        Returns:
            ndarray: Hip center [x, y]
        """
        left_hip = keypoints[LEFT_HIP, :2]
        right_hip = keypoints[RIGHT_HIP, :2]
        return (left_hip + right_hip) / 2.0
    
    def compute_shoulder_center(self, keypoints):
        """
        Compute shoulder center as midpoint of left and right shoulder.
        
        Parameters:
            keypoints: Shape (17, 2) or (17, 3)
            
        Returns:
            ndarray: Shoulder center [x, y]
        """
        left_shoulder = keypoints[LEFT_SHOULDER, :2]
        right_shoulder = keypoints[RIGHT_SHOULDER, :2]
        return (left_shoulder + right_shoulder) / 2.0
    
    def compute_torso_length(self, keypoints):
        """
        Compute torso length as distance from hip center to shoulder center.
        
        This provides a scale factor for normalization.
        
        Parameters:
            keypoints: Shape (17, 2) or (17, 3)
            
        Returns:
            float: Torso length
        """
        hip_center = self.compute_hip_center(keypoints)
        shoulder_center = self.compute_shoulder_center(keypoints)
        
        # Euclidean distance
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        # Avoid division by zero
        return max(torso_length, 1e-6)
    
    def normalize_frame(self, keypoints):
        """
        Normalize single frame of keypoints.
        
        Normalization formula (as per paper):
            P_normalized = (P_original - hip_center) / torso_length
        
        Parameters:
            keypoints: Shape (17, 3) with (x, y, confidence)
            
        Returns:
            ndarray: Normalized keypoints shape (17, 3)
        """
        normalized = keypoints.copy()
        
        # Step 1: Compute hip center
        self.hip_center = self.compute_hip_center(keypoints)
        
        # Step 2: Compute torso length
        self.torso_length = self.compute_torso_length(keypoints)
        
        # Step 3: Translate to center at hip
        normalized[:, 0] -= self.hip_center[0]  # x
        normalized[:, 1] -= self.hip_center[1]  # y
        
        # Step 4: Scale by torso length
        normalized[:, 0] /= self.torso_length
        normalized[:, 1] /= self.torso_length
        
        # Keep confidence unchanged
        return normalized
    
    def normalize_sequence(self, keypoints_sequence):
        """
        Normalize sequence of keypoints (video/repetition).
        
        Parameters:
            keypoints_sequence: Shape (frames, 17, 3)
            
        Returns:
            ndarray: Normalized sequence shape (frames, 17, 3)
        """
        normalized_sequence = np.zeros_like(keypoints_sequence)
        
        for i in range(len(keypoints_sequence)):
            normalized_sequence[i] = self.normalize_frame(keypoints_sequence[i])
        
        return normalized_sequence
    
    def denormalize_frame(self, normalized_keypoints, hip_center, torso_length):
        """
        Reverse normalization to get original scale coordinates.
        
        Parameters:
            normalized_keypoints: Normalized keypoints (17, 3)
            hip_center: Original hip center [x, y]
            torso_length: Original torso length
            
        Returns:
            ndarray: Denormalized keypoints (17, 3)
        """
        denormalized = normalized_keypoints.copy()
        
        # Reverse scale
        denormalized[:, 0] *= torso_length
        denormalized[:, 1] *= torso_length
        
        # Reverse translation
        denormalized[:, 0] += hip_center[0]
        denormalized[:, 1] += hip_center[1]
        
        return denormalized


def normalize_keypoints(keypoints):
    """
    Convenience function to normalize keypoints.
    
    Parameters:
        keypoints: Shape (17, 3) or (frames, 17, 3)
        
    Returns:
        ndarray: Normalized keypoints same shape as input
    """
    normalizer = SkeletonNormalizer()
    
    if keypoints.ndim == 2:
        return normalizer.normalize_frame(keypoints)
    elif keypoints.ndim == 3:
        return normalizer.normalize_sequence(keypoints)
    else:
        raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")


if __name__ == "__main__":
    # Test normalization
    print("Testing Skeleton Normalization Module...")
    
    # Create sample keypoints (17 joints, x, y, confidence)
    sample_keypoints = np.random.rand(17, 3)
    sample_keypoints[:, 2] = 1.0  # Set all confidence to 1
    
    normalizer = SkeletonNormalizer()
    normalized = normalizer.normalize_frame(sample_keypoints)
    
    print(f"Original hip center: {normalizer.hip_center}")
    print(f"Torso length: {normalizer.torso_length:.4f}")
    print(f"Original shape: {sample_keypoints.shape}")
    print(f"Normalized shape: {normalized.shape}")
    
    # Verify hip is now at origin
    new_hip_center = (normalized[LEFT_HIP, :2] + normalized[RIGHT_HIP, :2]) / 2
    print(f"Normalized hip center: {new_hip_center}")  # Should be ~[0, 0]
