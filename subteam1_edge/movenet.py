"""
Module 1: MoveNet Pose Estimation
---------------------------------
MoveNet Thunder-SinglePose implementation as described in research paper.

Research Paper Reference:
    "The MoveNet model was used to extract 17 keypoints from each frame.
    MoveNet Thunder variant provides higher accuracy for pose estimation.
    Input size: 256x256 pixels (Thunder variant)"

17 Keypoints (as per paper):
    0: nose          1: left_eye       2: right_eye
    3: left_ear      4: right_ear      5: left_shoulder
    6: right_shoulder 7: left_elbow    8: right_elbow
    9: left_wrist    10: right_wrist   11: left_hip
    12: right_hip    13: left_knee     14: right_knee
    15: left_ankle   16: right_ankle
"""

import numpy as np
import cv2
import os

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow tensorflow-hub")


# MoveNet keypoint names (17 total as per paper)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for visualization (as per paper - green for correct, blue for incorrect)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

# MoveNet URLs
MOVENET_THUNDER_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
MOVENET_LIGHTNING_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"


class MoveNetPoseEstimator:
    """
    MoveNet Thunder pose estimation as described in research paper.
    
    Paper specifications:
        - Model: MoveNet Thunder (SinglePose)
        - Input: 256x256 RGB image
        - Output: 17 keypoints with (y, x, confidence)
        - Confidence threshold: 0.8 for reliable detection
    """
    
    def __init__(self, use_lightning=False):
        """Initialize MoveNet model."""
        self.use_lightning = use_lightning
        # Thunder uses 256x256, Lightning uses 192x192
        self.input_size = 192 if use_lightning else 256  
        self.model = None
        self.movenet = None
        
        if TF_AVAILABLE:
            model_name = "Lightning (Fast)" if use_lightning else "Thunder (Accurate)"
            url = MOVENET_LIGHTNING_URL if use_lightning else MOVENET_THUNDER_URL
            print(f"Loading MoveNet {model_name} model...")
            try:
                self.model = hub.load(url)
                self.movenet = self.model.signatures['serving_default']
                print(f"MoveNet {model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading MoveNet: {e}")
                self.model = None
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for MoveNet input.
        
        Parameters:
            frame: BGR image from OpenCV
            
        Returns:
            Tensor: Preprocessed image tensor [1, 256, 256, 3]
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256 (Thunder input size)
        resized = cv2.resize(rgb_frame, (self.input_size, self.input_size))
        
        # Convert to tensor and add batch dimension
        input_tensor = tf.cast(resized, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def extract_keypoints(self, frame):
        """
        Extract 17 keypoints from a single frame.
        
        Parameters:
            frame: BGR image from OpenCV
            
        Returns:
            ndarray: Shape (17, 3) with (x, y, confidence) per keypoint
                     Coordinates normalized to [0, 1]
        """
        if self.movenet is None:
            # Return zeros if model not available
            return np.zeros((17, 3))
        
        # Preprocess
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        # MoveNet outputs (y, x, confidence), convert to (x, y, confidence)
        result = np.zeros((17, 3))
        result[:, 0] = keypoints[:, 1]  # x
        result[:, 1] = keypoints[:, 0]  # y
        result[:, 2] = keypoints[:, 2]  # confidence
        
        return result
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process entire video and extract keypoints for all frames.
        
        Parameters:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)
            
        Returns:
            ndarray: Shape (num_frames, 17, 3)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        all_keypoints = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                keypoints = self.extract_keypoints(frame)
                all_keypoints.append(keypoints)
            
            frame_idx += 1
        
        cap.release()
        return np.array(all_keypoints)
    
    def visualize_skeleton(self, frame, keypoints, is_correct=True, confidence_threshold=0.8):
        """
        Draw skeleton on frame with color coding (as per paper).
        
        Paper specification:
            - Green skeleton: Correct execution
            - Blue skeleton: Incorrect execution
        
        Parameters:
            frame: BGR image
            keypoints: Shape (17, 3) with (x, y, confidence)
            is_correct: True for green, False for blue
            confidence_threshold: Minimum confidence to draw (0.8 as per paper)
            
        Returns:
            frame: Annotated image
        """
        h, w = frame.shape[:2]
        
        # Color coding as per paper
        color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green or Blue (BGR)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= confidence_threshold:
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 5, color, -1)
        
        # Draw skeleton connections
        for i, j in SKELETON_CONNECTIONS:
            if keypoints[i, 2] >= confidence_threshold and keypoints[j, 2] >= confidence_threshold:
                x1, y1 = int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)
                x2, y2 = int(keypoints[j, 0] * w), int(keypoints[j, 1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        return frame


def load_dataset_keypoints(dataset_path, exercise_id, video_id, camera="c17", fps=30):
    """
    Load pre-extracted keypoints from REHAB24-6 dataset.
    
    Note: Dataset provides 26 joints, but we use first 17 to match MoveNet format.
    
    Parameters:
        dataset_path: Path to dataset root
        exercise_id: Exercise folder (Ex1, Ex2, etc.)
        video_id: Video identifier (PM_000, PM_001, etc.)
        camera: Camera view (c17 or c18)
        fps: Frame rate (30 or 120)
        
    Returns:
        ndarray: Shape (frames, 17, 2) - only x, y coordinates
    """
    filename = f"{video_id}-{camera}-{fps}fps.npy"
    filepath = os.path.join(dataset_path, "2d_joints", exercise_id, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Keypoints file not found: {filepath}")
    
    # Load data - shape is (frames, joints, coords)
    data = np.load(filepath)
    
    # Dataset has 26 joints, we need to map to 17 MoveNet keypoints
    # REHAB24-6 joint mapping to MoveNet format:
    # We'll use a subset that matches MoveNet structure
    # For now, use first 17 joints and add confidence=1.0
    
    if data.shape[1] >= 17:
        keypoints_17 = data[:, :17, :2]  # Take first 17 joints, x and y only
    else:
        # Pad if needed
        keypoints_17 = np.zeros((data.shape[0], 17, 2))
        keypoints_17[:, :data.shape[1], :] = data[:, :, :2]
    
    # Add confidence scores (1.0 for all since these are ground truth)
    confidence = np.ones((keypoints_17.shape[0], 17, 1))
    keypoints_with_conf = np.concatenate([keypoints_17, confidence], axis=2)
    
    return keypoints_with_conf


if __name__ == "__main__":
    # Test MoveNet loading
    print("Testing MoveNet Pose Estimation Module...")
    
    if TF_AVAILABLE:
        estimator = MoveNetPoseEstimator()
        print(f"Model loaded: {estimator.model is not None}")
        print(f"Input size: {estimator.input_size}x{estimator.input_size}")
        print(f"Number of keypoints: {len(KEYPOINT_NAMES)}")
        print(f"Keypoints: {KEYPOINT_NAMES}")
    else:
        print("TensorFlow not available. Please install: pip install tensorflow tensorflow-hub")
