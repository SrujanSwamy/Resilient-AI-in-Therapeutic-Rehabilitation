"""
Module 5: Confidence Scoring
----------------------------
Implements confidence scoring mechanism as per research paper.

Research Paper Reference:
    "A threshold of 0.8 is established: if the confidence score exceeds this value,
    the estimated movement is considered reliable and can be evaluated further.
    If the confidence score falls below this threshold, the system displays a warning."

Confidence Computation:
    confidence = 1 / (1 + RMSE)
    - Normalized to range [0, 1]
    - Higher values = more reliable predictions
"""

import numpy as np
from typing import Tuple, Dict


# Research paper confidence threshold
CONFIDENCE_THRESHOLD = 0.8


def compute_confidence_from_rmse(rmse: float, scale_factor: float = 1.0) -> float:
    """
    Convert RMSE to confidence score.
    
    Paper formula: confidence = 1 / (1 + RMSE/scale_factor)
    
    Parameters:
        rmse: Root mean squared error value
        scale_factor: Normalization factor
        
    Returns:
        float: Confidence score in range (0, 1]
    """
    return 1.0 / (1.0 + rmse / scale_factor)


def check_confidence_threshold(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> Tuple[bool, str]:
    """
    Check if confidence meets reliability threshold.
    
    As per paper:
    - >= 0.8: Reliable, proceed with classification
    - < 0.8: Unreliable, display warning
    
    Parameters:
        confidence: Computed confidence score
        threshold: Reliability threshold (default 0.8)
        
    Returns:
        tuple: (is_reliable, message)
    """
    if confidence >= threshold:
        return True, f"Reliable (confidence={confidence:.2f})"
    else:
        return False, f"Warning: Low confidence ({confidence:.2f}). Pose detection may be unreliable."


class ConfidenceScorer:
    """
    Confidence scoring system as per research paper.
    
    Attributes:
        threshold: Confidence threshold (0.8 as per paper)
        scale_factor: RMSE scaling for confidence computation
    """
    
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD, scale_factor: float = 0.1):
        """
        Initialize confidence scorer.
        
        Parameters:
            threshold: Reliability threshold (0.8)
            scale_factor: RMSE scale factor for confidence conversion
        """
        self.threshold = threshold
        self.scale_factor = scale_factor
    
    def score_from_rmse(self, rmse: float) -> float:
        """
        Compute confidence from RMSE.
        
        Parameters:
            rmse: Root mean squared error
            
        Returns:
            float: Confidence score [0, 1]
        """
        return compute_confidence_from_rmse(rmse, self.scale_factor)
    
    def score_from_keypoint_confidence(self, keypoints: np.ndarray) -> float:
        """
        Compute confidence from MoveNet keypoint confidence scores.
        
        The average confidence across all 17 keypoints.
        
        Parameters:
            keypoints: Shape (17, 3) with (x, y, confidence)
            
        Returns:
            float: Mean keypoint confidence
        """
        return np.mean(keypoints[:, 2])
    
    def is_reliable(self, confidence: float) -> bool:
        """
        Check if prediction is reliable.
        
        Parameters:
            confidence: Confidence score
            
        Returns:
            bool: True if confidence >= threshold
        """
        return confidence >= self.threshold
    
    def evaluate(self, keypoints: np.ndarray, rmse: float) -> Dict:
        """
        Full evaluation of reliability.
        
        Parameters:
            keypoints: Keypoints with confidence (17, 3)
            rmse: Comparison RMSE
            
        Returns:
            dict: Reliability assessment
        """
        keypoint_confidence = self.score_from_keypoint_confidence(keypoints)
        comparison_confidence = self.score_from_rmse(rmse)
        
        # Combined confidence (average of both)
        combined_confidence = (keypoint_confidence + comparison_confidence) / 2
        
        is_reliable = self.is_reliable(combined_confidence)
        
        return {
            'keypoint_confidence': keypoint_confidence,
            'comparison_confidence': comparison_confidence,
            'combined_confidence': combined_confidence,
            'is_reliable': is_reliable,
            'threshold': self.threshold,
            'message': "Reliable" if is_reliable else "Warning: Low confidence"
        }


if __name__ == "__main__":
    # Test confidence scoring
    print("Testing Confidence Scoring Module...")
    
    scorer = ConfidenceScorer(threshold=0.8)
    
    # Test RMSE to confidence
    test_rmse_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    print("\nRMSE to Confidence conversion:")
    for rmse in test_rmse_values:
        conf = scorer.score_from_rmse(rmse)
        reliable = scorer.is_reliable(conf)
        print(f"  RMSE={rmse:.2f} -> Confidence={conf:.3f} -> {'Reliable' if reliable else 'Warning'}")
