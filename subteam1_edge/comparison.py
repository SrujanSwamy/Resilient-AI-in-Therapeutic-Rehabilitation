"""
Module 4: Movement Comparison
-----------------------------
Computes deviation metrics between observed movement and reference template.
Implements exact formulas from research paper.

Research Paper Formulas:

1. Euclidean Distance (Correction Vector):
    C = ||Y_correct - Y_actual|| = sqrt(sum((y_correct,i - y_actual,i)^2))
    
    where:
        Y_correct = Reference keypoint coordinates
        Y_actual = Observed keypoint coordinates
        C = Overall deviation measure

2. Mean Squared Error (MSE):
    MSE = (1/N) * sum((y_correct,i - y_actual,i)^2)

3. Root Mean Squared Error (RMSE):
    RMSE = sqrt(MSE)

Interpretation (as per paper):
    - Low C/RMSE: Movement matches reference (likely correct)
    - High C/RMSE: Significant deviation (likely incorrect)
"""

import numpy as np
from typing import Tuple, Dict


def compute_euclidean_distance(reference: np.ndarray, observed: np.ndarray) -> float:
    """
    Compute Euclidean distance between reference and observed positions.
    
    Paper formula: C = ||Y_correct - Y_actual||
    
    Parameters:
        reference: Reference keypoints shape (17, 2) or (frames, 17, 2)
        observed: Observed keypoints shape (17, 2) or (frames, 17, 2)
        
    Returns:
        float: L2 norm of the difference
    """
    diff = reference.flatten() - observed.flatten()
    return np.linalg.norm(diff)


def compute_per_joint_distance(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance for each joint independently.
    
    Identifies which joints deviate most from reference.
    
    Parameters:
        reference: Reference keypoints shape (17, 2)
        observed: Observed keypoints shape (17, 2)
        
    Returns:
        ndarray: Distance per joint shape (17,)
    """
    return np.sqrt(np.sum((reference - observed) ** 2, axis=1))


def compute_mse(reference: np.ndarray, observed: np.ndarray) -> float:
    """
    Compute Mean Squared Error as per paper.
    
    Paper formula: MSE = (1/N) * sum((y_correct,i - y_actual,i)^2)
    
    Parameters:
        reference: Reference keypoints
        observed: Observed keypoints
        
    Returns:
        float: Mean squared error
    """
    diff = reference - observed
    return np.mean(diff ** 2)


def compute_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error as per paper.
    
    Paper formula: RMSE = sqrt(MSE)
    
    Parameters:
        reference: Reference keypoints
        observed: Observed keypoints
        
    Returns:
        float: Root mean squared error
    """
    return np.sqrt(compute_mse(reference, observed))


class MovementComparator:
    """
    Compares observed movement against reference template.
    
    As per research paper:
    - Computes frame-by-frame deviation
    - Aggregates metrics over entire repetition
    - Identifies most deviant joints
    
    Attributes:
        reference_mean: Reference template from Module 3
        reference_std: Reference variability bounds
    """
    
    def __init__(self, reference_mean: np.ndarray, reference_std: np.ndarray = None):
        """
        Initialize comparator with reference model.
        
        Parameters:
            reference_mean: Mean keypoints shape (frames, 17, 2)
            reference_std: Standard deviation shape (frames, 17, 2) [optional]
        """
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.target_frames = len(reference_mean)
    
    def compare_frame(self, frame_idx: int, observed: np.ndarray) -> Dict:
        """
        Compare single frame against reference.
        
        Parameters:
            frame_idx: Reference frame index
            observed: Observed keypoints shape (17, 2)
            
        Returns:
            dict: Deviation metrics for this frame
        """
        reference = self.reference_mean[frame_idx]
        
        euclidean = compute_euclidean_distance(reference, observed)
        mse = compute_mse(reference, observed)
        rmse = compute_rmse(reference, observed)
        per_joint = compute_per_joint_distance(reference, observed)
        
        return {
            'euclidean_distance': euclidean,
            'mse': mse,
            'rmse': rmse,
            'per_joint_distance': per_joint,
            'max_joint_distance': np.max(per_joint),
            'max_joint_idx': int(np.argmax(per_joint))
        }
    
    def compare_sequence(self, observed_sequence: np.ndarray) -> Dict:
        """
        Compare entire observed sequence against reference.
        
        Parameters:
            observed_sequence: Observed keypoints shape (frames, 17, 2)
            
        Returns:
            dict: Aggregated metrics over all frames
        """
        # Ensure same length as reference
        if len(observed_sequence) != self.target_frames:
            observed_sequence = self._resample(observed_sequence, self.target_frames)
        
        frame_metrics = []
        all_euclidean = []
        all_rmse = []
        all_per_joint = []
        
        for i in range(self.target_frames):
            metrics = self.compare_frame(i, observed_sequence[i])
            frame_metrics.append(metrics)
            all_euclidean.append(metrics['euclidean_distance'])
            all_rmse.append(metrics['rmse'])
            all_per_joint.append(metrics['per_joint_distance'])
        
        all_per_joint = np.array(all_per_joint)  # (frames, 17)
        
        # Aggregate statistics
        return {
            'mean_euclidean': np.mean(all_euclidean),
            'max_euclidean': np.max(all_euclidean),
            'mean_rmse': np.mean(all_rmse),
            'max_rmse': np.max(all_rmse),
            'overall_mse': compute_mse(self.reference_mean, observed_sequence),
            'overall_rmse': compute_rmse(self.reference_mean, observed_sequence),
            'per_joint_mean': np.mean(all_per_joint, axis=0),  # (17,)
            'worst_frame_idx': int(np.argmax(all_rmse)),
            'worst_joint_idx': int(np.argmax(np.mean(all_per_joint, axis=0))),
            'frame_metrics': frame_metrics
        }
    
    def _resample(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample sequence to target length.
        
        Parameters:
            sequence: Input sequence shape (frames, 17, 2)
            target_length: Desired length
            
        Returns:
            ndarray: Resampled sequence
        """
        current_length = len(sequence)
        indices = np.linspace(0, current_length - 1, target_length)
        
        resampled = np.zeros((target_length, 17, 2))
        
        for i in range(target_length):
            idx = indices[i]
            lower = int(np.floor(idx))
            upper = min(lower + 1, current_length - 1)
            weight = idx - lower
            resampled[i] = (1 - weight) * sequence[lower] + weight * sequence[upper]
        
        return resampled
    
    def get_deviation_summary(self, metrics: Dict) -> str:
        """
        Generate human-readable deviation summary.
        
        Parameters:
            metrics: Output from compare_sequence()
            
        Returns:
            str: Summary text
        """
        joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        worst_joint = joint_names[metrics['worst_joint_idx']]
        
        summary = f"""
Movement Comparison Summary:
---------------------------
Overall RMSE: {metrics['overall_rmse']:.4f}
Mean Frame RMSE: {metrics['mean_rmse']:.4f}
Max Frame RMSE: {metrics['max_rmse']:.4f}
Worst Frame: {metrics['worst_frame_idx']}
Most Deviant Joint: {worst_joint} (idx={metrics['worst_joint_idx']})
"""
        return summary


if __name__ == "__main__":
    # Test comparison module
    print("Testing Movement Comparison Module...")
    
    # Create sample reference and observed data
    np.random.seed(42)
    num_frames = 50
    
    # Reference template (simulated)
    reference_mean = np.random.rand(num_frames, 17, 2)
    
    # Observed with small deviation (should be "correct")
    observed_correct = reference_mean + np.random.normal(0, 0.01, reference_mean.shape)
    
    # Observed with large deviation (should be "incorrect")
    observed_incorrect = reference_mean + np.random.normal(0, 0.1, reference_mean.shape)
    
    comparator = MovementComparator(reference_mean)
    
    # Compare correct
    metrics_correct = comparator.compare_sequence(observed_correct)
    print(f"\nCorrect Sample:")
    print(f"  Overall RMSE: {metrics_correct['overall_rmse']:.4f}")
    print(f"  Mean RMSE: {metrics_correct['mean_rmse']:.4f}")
    
    # Compare incorrect
    metrics_incorrect = comparator.compare_sequence(observed_incorrect)
    print(f"\nIncorrect Sample:")
    print(f"  Overall RMSE: {metrics_incorrect['overall_rmse']:.4f}")
    print(f"  Mean RMSE: {metrics_incorrect['mean_rmse']:.4f}")
