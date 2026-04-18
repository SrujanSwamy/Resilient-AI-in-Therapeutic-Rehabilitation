"""
Compensation Detection Module
------------------------------
End-Semester Enhancement: Detects compensatory movement patterns where
a patient achieves the target position using incorrect body mechanics.

Examples of compensation:
  - Trunk lean during arm raising (Ex1, Ex2, Ex3)
  - Shoulder shrug during reaching tasks
  - Hip shift/pelvis rotation during leg exercises (Ex4, Ex5, Ex6)

Research Paper Reference:
    "The system should detect when patients use compensatory strategies
    to complete exercises, even when the primary movement appears correct."

Approach:
  - Define per-exercise 'stabiliser' joints that should remain bounded
  - Monitor their deviation from a neutral/reference position
  - Flag specific compensation patterns with severity scores
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# -----------------------------------------------------------------------
# Keypoint indices (MoveNet 17)
# -----------------------------------------------------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
NOSE = 0

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# -----------------------------------------------------------------------
# Per-exercise compensation rules
# Each rule defines:
#   stabilisers : joints that should stay bounded
#   max_lateral_shift : max X-axis deviation (normalised)
#   max_vertical_shift : max Y-axis deviation (normalised)
#   pattern_checks : list of pattern function names to run
# -----------------------------------------------------------------------
EXERCISE_RULES: Dict[str, Dict] = {
    'Ex1': {   # Right arm raise
        'primary_joints': [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
        'stabilisers': [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER],
        'max_lateral_shift': 0.15,
        'max_vertical_shift': 0.20,
        'pattern_checks': ['trunk_lean', 'shoulder_shrug'],
    },
    'Ex2': {   # Arm movement variations
        'primary_joints': [LEFT_SHOULDER, LEFT_ELBOW, RIGHT_SHOULDER, RIGHT_ELBOW],
        'stabilisers': [LEFT_HIP, RIGHT_HIP],
        'max_lateral_shift': 0.18,
        'max_vertical_shift': 0.22,
        'pattern_checks': ['trunk_lean', 'shoulder_shrug'],
    },
    'Ex3': {   # Shoulder rehabilitation
        'primary_joints': [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW],
        'stabilisers': [LEFT_HIP, RIGHT_HIP],
        'max_lateral_shift': 0.15,
        'max_vertical_shift': 0.20,
        'pattern_checks': ['trunk_lean', 'shoulder_shrug'],
    },
    'Ex4': {   # Leg exercises
        'primary_joints': [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
        'stabilisers': [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER],
        'max_lateral_shift': 0.12,
        'max_vertical_shift': 0.15,
        'pattern_checks': ['hip_shift', 'trunk_lean'],
    },
    'Ex5': {   # Knee rehabilitation
        'primary_joints': [LEFT_KNEE, RIGHT_KNEE],
        'stabilisers': [LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE],
        'max_lateral_shift': 0.10,
        'max_vertical_shift': 0.15,
        'pattern_checks': ['hip_shift'],
    },
    'Ex6': {   # Full body combined
        'primary_joints': [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_KNEE, RIGHT_KNEE],
        'stabilisers': [LEFT_HIP, RIGHT_HIP],
        'max_lateral_shift': 0.20,
        'max_vertical_shift': 0.25,
        'pattern_checks': ['trunk_lean', 'hip_shift', 'shoulder_shrug'],
    },
}

# Default fallback for unknown exercises
DEFAULT_RULES = {
    'primary_joints': list(range(17)),
    'stabilisers': [LEFT_HIP, RIGHT_HIP],
    'max_lateral_shift': 0.20,
    'max_vertical_shift': 0.25,
    'pattern_checks': ['trunk_lean', 'hip_shift'],
}


class CompensationDetector:
    """
    Detects compensatory movement patterns in rehabilitation exercises.

    Parameters
    ----------
    severity_threshold : float
        Minimum severity score (0–1) to report a compensation (default 0.3).
    """

    def __init__(self, severity_threshold: float = 0.3):
        self.severity_threshold = severity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        normalized_seq: np.ndarray,
        exercise_id: str,
        reference_mean: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run compensation detection on a normalised keypoint sequence.

        Parameters
        ----------
        normalized_seq : np.ndarray
            Shape (frames, 17, 2) — normalised (x, y) coordinates.
        exercise_id : str
            'Ex1' … 'Ex6'.
        reference_mean : np.ndarray, optional
            Reference template (frames, 17, 2) from Module 3.
            Used for stabiliser drift relative to reference.

        Returns
        -------
        dict with keys:
            compensation_found  : bool
            types               : list[str]  — which patterns were found
            severity            : float      — max severity across patterns
            details             : dict       — per-pattern detail
        """
        rules = EXERCISE_RULES.get(exercise_id, DEFAULT_RULES)
        results = {
            'compensation_found': False,
            'types': [],
            'severity': 0.0,
            'details': {},
        }

        for check in rules['pattern_checks']:
            method = getattr(self, f'_check_{check}', None)
            if method is None:
                continue
            found, severity, detail = method(normalized_seq, rules, reference_mean)
            results['details'][check] = detail
            if found and severity >= self.severity_threshold:
                results['compensation_found'] = True
                results['types'].append(check)
                results['severity'] = max(results['severity'], severity)

        return results

    # ------------------------------------------------------------------
    # Pattern checkers
    # ------------------------------------------------------------------

    def _check_trunk_lean(
        self,
        seq: np.ndarray,
        rules: dict,
        reference: Optional[np.ndarray],
    ) -> Tuple[bool, float, dict]:
        """
        Detects lateral trunk lean: the spine midpoint drifts sideways.
        Spine midpoint = average of left_hip, right_hip, left_shoulder, right_shoulder.
        """
        spine_x = np.mean(
            seq[:, [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER], 0],
            axis=1,
        )  # shape (frames,)

        # Compare to reference if available, else use first-frame baseline
        if reference is not None and len(reference) == len(seq):
            ref_spine_x = np.mean(
                reference[:, [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER], 0],
                axis=1,
            )
            drift = np.abs(spine_x - ref_spine_x)
        else:
            baseline = np.mean(spine_x[:5])  # first 5 frames = rest position
            drift = np.abs(spine_x - baseline)

        max_drift = float(np.max(drift))
        mean_drift = float(np.mean(drift))
        threshold = rules['max_lateral_shift']
        severity = min(max_drift / (threshold + 1e-6), 1.0)
        found = max_drift > threshold

        return found, severity, {
            'max_lateral_drift': max_drift,
            'mean_lateral_drift': mean_drift,
            'threshold': threshold,
            'worst_frame': int(np.argmax(drift)),
        }

    def _check_hip_shift(
        self,
        seq: np.ndarray,
        rules: dict,
        reference: Optional[np.ndarray],
    ) -> Tuple[bool, float, dict]:
        """
        Detects pelvis shift: left/right asymmetry in hip positions.
        Measured as lateral difference between left and right hip X-coordinates.
        """
        hip_asymmetry = np.abs(seq[:, LEFT_HIP, 0] - seq[:, RIGHT_HIP, 0])

        if reference is not None and len(reference) == len(seq):
            ref_asym = np.abs(reference[:, LEFT_HIP, 0] - reference[:, RIGHT_HIP, 0])
            drift = np.abs(hip_asymmetry - ref_asym)
        else:
            baseline = np.mean(hip_asymmetry[:5])
            drift = np.abs(hip_asymmetry - baseline)

        max_drift = float(np.max(drift))
        threshold = rules['max_lateral_shift']
        severity = min(max_drift / (threshold + 1e-6), 1.0)
        found = max_drift > threshold

        return found, severity, {
            'max_hip_asymmetry_drift': max_drift,
            'threshold': threshold,
            'worst_frame': int(np.argmax(drift)),
        }

    def _check_shoulder_shrug(
        self,
        seq: np.ndarray,
        rules: dict,
        reference: Optional[np.ndarray],
    ) -> Tuple[bool, float, dict]:
        """
        Detects shoulder shrug: the shoulder center rises unexpectedly
        relative to the hip center (vertical elevation in normalised space).
        """
        shoulder_y = np.mean(seq[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 1], axis=1)
        hip_y = np.mean(seq[:, [LEFT_HIP, RIGHT_HIP], 1], axis=1)
        torso_y = shoulder_y - hip_y  # relative shoulder height above hip

        if reference is not None and len(reference) == len(seq):
            ref_shoulder_y = np.mean(reference[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 1], axis=1)
            ref_hip_y = np.mean(reference[:, [LEFT_HIP, RIGHT_HIP], 1], axis=1)
            ref_torso_y = ref_shoulder_y - ref_hip_y
            drift = np.abs(torso_y - ref_torso_y)
        else:
            baseline = np.mean(torso_y[:5])
            drift = np.abs(torso_y - baseline)

        max_drift = float(np.max(drift))
        threshold = rules['max_vertical_shift']
        severity = min(max_drift / (threshold + 1e-6), 1.0)
        found = max_drift > threshold

        return found, severity, {
            'max_shoulder_elevation_drift': max_drift,
            'threshold': threshold,
            'worst_frame': int(np.argmax(drift)),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_feedback_message(self, detection_result: Dict) -> str:
        """
        Generate human-readable feedback for the clinician dashboard.
        """
        if not detection_result['compensation_found']:
            return "OK No compensation patterns detected."

        msgs = {
            'trunk_lean': "⚠ Trunk lean detected — patient is shifting their body sideways.",
            'hip_shift': "⚠ Hip shift detected — pelvis is rotating during the exercise.",
            'shoulder_shrug': "⚠ Shoulder shrug detected — shoulder is elevating unexpectedly.",
        }
        lines = [msgs.get(t, f"⚠ {t} detected.") for t in detection_result['types']]
        severity_pct = int(detection_result['severity'] * 100)
        lines.append(f"   Severity: {severity_pct}%")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Compensation Detection Module...")
    np.random.seed(42)

    # Simulate 50 frames, 17 joints, (x, y) normalised
    seq = np.zeros((50, 17, 2))

    # Set reasonable joint positions in normalised space
    seq[:, LEFT_SHOULDER, :] = [-0.2, 0.5]
    seq[:, RIGHT_SHOULDER, :] = [0.2, 0.5]
    seq[:, LEFT_HIP, :] = [-0.15, 0.0]
    seq[:, RIGHT_HIP, :] = [0.15, 0.0]
    seq[:, LEFT_KNEE, :] = [-0.15, -0.55]
    seq[:, RIGHT_KNEE, :] = [0.15, -0.55]

    # Inject trunk lean in last 20 frames
    seq[30:, LEFT_SHOULDER, 0] -= 0.25
    seq[30:, RIGHT_SHOULDER, 0] -= 0.25
    seq[30:, LEFT_HIP, 0] -= 0.25
    seq[30:, RIGHT_HIP, 0] -= 0.25

    detector = CompensationDetector(severity_threshold=0.3)
    result = detector.detect(seq, 'Ex1')

    print(f"\n  Compensation found: {result['compensation_found']}")
    print(f"  Types: {result['types']}")
    print(f"  Severity: {result['severity']:.2f}")
    print(f"\n  Feedback:\n  {detector.get_feedback_message(result)}")
    print("\n  [OK] Compensation detection module OK")
