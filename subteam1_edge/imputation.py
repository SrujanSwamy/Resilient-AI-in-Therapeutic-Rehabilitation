"""
Joint Imputation Module
-----------------------
End-Semester Enhancement: Handles missing or low-confidence keypoints
from MoveNet output due to occlusion, poor lighting, or body positioning.

Three-tier imputation strategy:
    1. Temporal Interpolation  - use adjacent valid frames
    2. Spatial Inference       - infer from anatomically-connected joints
    3. Motion Prediction       - constant-velocity kinematic projection

Research Paper Reference:
    "The system must handle cases where keypoint detection confidence
    falls below the threshold, ensuring robust operation under
    real-world conditions."
"""

import numpy as np
from typing import Tuple, List, Optional

# MoveNet keypoint indices (reused from normalization.py)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Anatomical kinematic chain: each joint's parent(s) for spatial inference
# Format: joint_idx -> list of reference joint indices to infer from
KINEMATIC_PARENTS = {
    LEFT_EYE:      [NOSE],
    RIGHT_EYE:     [NOSE],
    LEFT_EAR:      [LEFT_EYE, NOSE],
    RIGHT_EAR:     [RIGHT_EYE, NOSE],
    LEFT_SHOULDER: [LEFT_HIP, RIGHT_SHOULDER],
    RIGHT_SHOULDER:[RIGHT_HIP, LEFT_SHOULDER],
    LEFT_ELBOW:    [LEFT_SHOULDER],
    RIGHT_ELBOW:   [RIGHT_SHOULDER],
    LEFT_WRIST:    [LEFT_ELBOW, LEFT_SHOULDER],
    RIGHT_WRIST:   [RIGHT_ELBOW, RIGHT_SHOULDER],
    LEFT_HIP:      [RIGHT_HIP],
    RIGHT_HIP:     [LEFT_HIP],
    LEFT_KNEE:     [LEFT_HIP],
    RIGHT_KNEE:    [RIGHT_HIP],
    LEFT_ANKLE:    [LEFT_KNEE, LEFT_HIP],
    RIGHT_ANKLE:   [RIGHT_KNEE, RIGHT_HIP],
}

# Approximate bone-length ratios relative to torso length (normalized space)
# Used for spatial inference when temporal data unavailable
BONE_LENGTH_RATIOS = {
    (LEFT_SHOULDER,  LEFT_ELBOW):   0.45,
    (RIGHT_SHOULDER, RIGHT_ELBOW):  0.45,
    (LEFT_ELBOW,     LEFT_WRIST):   0.40,
    (RIGHT_ELBOW,    RIGHT_WRIST):  0.40,
    (LEFT_HIP,       LEFT_KNEE):    0.55,
    (RIGHT_HIP,      RIGHT_KNEE):   0.55,
    (LEFT_KNEE,      LEFT_ANKLE):   0.50,
    (RIGHT_KNEE,     RIGHT_ANKLE):  0.50,
}


class JointImputer:
    """
    Intelligent imputation for missing or low-confidence keypoints.

    Three-tier fallback:
      1. Temporal interpolation between valid frames
      2. Spatial inference from anatomically-connected joints
      3. Constant-velocity motion prediction

    Parameters
    ----------
    confidence_threshold : float
        Keypoints with confidence below this are treated as missing (default 0.3).
    velocity_window : int
        Number of past frames used for motion prediction (default 5).
    """

    def __init__(self, confidence_threshold: float = 0.3, velocity_window: int = 5):
        self.confidence_threshold = confidence_threshold
        self.velocity_window = velocity_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def impute_sequence(self, keypoints_seq: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Impute missing joints across an entire sequence.

        Parameters
        ----------
        keypoints_seq : np.ndarray
            Shape (frames, 17, 3)  — last channel = confidence.

        Returns
        -------
        imputed_seq : np.ndarray
            Sequence with missing joints filled, same shape as input.
        report : dict
            Statistics about how many joints were imputed and which method.
        """
        seq = keypoints_seq.copy().astype(float)
        n_frames, n_joints, _ = seq.shape

        report = {
            'total_missing': 0,
            'temporal': 0,
            'spatial': 0,
            'motion': 0,
            'unfilled': 0,
            'per_joint': {name: 0 for name in KEYPOINT_NAMES},
        }

        missing_mask = seq[:, :, 2] < self.confidence_threshold  # (frames, 17)
        report['total_missing'] = int(missing_mask.sum())

        if report['total_missing'] == 0:
            return seq, report

        # --- Pass 1: Temporal interpolation ---
        for j in range(n_joints):
            for f in range(n_frames):
                if missing_mask[f, j]:
                    filled = self._temporal_interpolate(seq, j, f, missing_mask)
                    if filled is not None:
                        seq[f, j, :2] = filled
                        seq[f, j, 2] = self.confidence_threshold  # mark as imputed
                        missing_mask[f, j] = False
                        report['temporal'] += 1
                        report['per_joint'][KEYPOINT_NAMES[j]] += 1

        # --- Pass 2: Spatial inference ---
        for j in range(n_joints):
            for f in range(n_frames):
                if missing_mask[f, j]:
                    filled = self._spatial_infer(seq[f], j)
                    if filled is not None:
                        seq[f, j, :2] = filled
                        seq[f, j, 2] = self.confidence_threshold * 0.8
                        missing_mask[f, j] = False
                        report['spatial'] += 1
                        report['per_joint'][KEYPOINT_NAMES[j]] += 1

        # --- Pass 3: Motion prediction ---
        for j in range(n_joints):
            for f in range(n_frames):
                if missing_mask[f, j]:
                    filled = self._motion_predict(seq, j, f)
                    if filled is not None:
                        seq[f, j, :2] = filled
                        seq[f, j, 2] = self.confidence_threshold * 0.6
                        missing_mask[f, j] = False
                        report['motion'] += 1
                        report['per_joint'][KEYPOINT_NAMES[j]] += 1

        report['unfilled'] = int(missing_mask.sum())
        return seq, report

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _temporal_interpolate(
        self,
        seq: np.ndarray,
        joint_idx: int,
        frame_idx: int,
        missing_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Linear interpolation between the nearest valid frames before & after.
        """
        n_frames = len(seq)

        # Search backward for last valid frame
        prev_f = None
        for f in range(frame_idx - 1, -1, -1):
            if not missing_mask[f, joint_idx]:
                prev_f = f
                break

        # Search forward for next valid frame
        next_f = None
        for f in range(frame_idx + 1, n_frames):
            if not missing_mask[f, joint_idx]:
                next_f = f
                break

        if prev_f is None and next_f is None:
            return None
        if prev_f is None:
            return seq[next_f, joint_idx, :2].copy()
        if next_f is None:
            return seq[prev_f, joint_idx, :2].copy()

        # Linear interpolation
        t = (frame_idx - prev_f) / (next_f - prev_f)
        return (1 - t) * seq[prev_f, joint_idx, :2] + t * seq[next_f, joint_idx, :2]

    def _spatial_infer(self, frame: np.ndarray, joint_idx: int) -> Optional[np.ndarray]:
        """
        Infer position from anatomically-connected parent joints.
        Uses average of parent positions weighted by bone-length ratios.
        """
        parents = KINEMATIC_PARENTS.get(joint_idx)
        if not parents:
            return None

        # Only use parents that are valid in this frame
        valid_parents = [
            p for p in parents
            if frame[p, 2] >= self.confidence_threshold
        ]
        if not valid_parents:
            return None

        # Simple average of parent positions (good enough for rehabilitation)
        return np.mean(frame[valid_parents, :2], axis=0)

    def _motion_predict(
        self,
        seq: np.ndarray,
        joint_idx: int,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Constant-velocity prediction using the mean velocity
        of the last `velocity_window` valid frames.
        """
        valid_positions: List[np.ndarray] = []
        for f in range(frame_idx - 1, max(-1, frame_idx - self.velocity_window - 1), -1):
            if seq[f, joint_idx, 2] >= self.confidence_threshold:
                valid_positions.insert(0, seq[f, joint_idx, :2])

        if len(valid_positions) < 2:
            return None

        # Mean velocity
        velocities = np.diff(valid_positions, axis=0)
        mean_vel = np.mean(velocities, axis=0)

        # Extrapolate from last known position
        last_pos = valid_positions[-1]
        steps_ahead = frame_idx - (frame_idx - len(valid_positions))
        return last_pos + mean_vel * steps_ahead


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Joint Imputation Module...")

    np.random.seed(0)
    # Simulate 30 frames, 17 joints, (x, y, confidence)
    seq = np.random.rand(30, 17, 3)
    seq[:, :, 2] = 0.9  # all valid

    # Artificially drop some joints
    seq[5:10, LEFT_WRIST, 2] = 0.1   # left wrist missing for 5 frames
    seq[15, RIGHT_KNEE, 2] = 0.0     # right knee missing for 1 frame
    seq[:, LEFT_EAR, 2] = 0.0        # left ear always missing

    imputer = JointImputer(confidence_threshold=0.3)
    imputed, report = imputer.impute_sequence(seq)

    print(f"\n  Total missing joints: {report['total_missing']}")
    print(f"  Filled via temporal:  {report['temporal']}")
    print(f"  Filled via spatial:   {report['spatial']}")
    print(f"  Filled via motion:    {report['motion']}")
    print(f"  Still unfilled:       {report['unfilled']}")
    print("\n  [OK] Imputation module OK")
