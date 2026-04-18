"""
Live Webcam Mode — Real-Time Exercise Feedback for Patients
============================================================
Open the webcam, detect pose with MoveNet, and use a Kinematic
State Machine to automatically segment exercise repetitions.

State Machine:
  IDLE       -> Waiting for patient to begin movement
  RECORDING  -> Patient is moving; accumulating frames
  ANALYSING  -> Movement stopped; running the AI pipeline
  COOLDOWN   -> Showing result; resting before next rep

Feedback shown to the patient:
  - Skeleton overlay (GREEN = correct, RED = incorrect, YELLOW = collecting)
  - Large verdict banner: CORRECT / INCORRECT / IDLE / RECORDING
  - RMSE similarity bar
  - Confidence bar
  - Fluidity score
  - Compensation warnings
  - State machine status indicator
  - Audio voice feedback (async TTS)
  - Press 'q' to quit | 's' to save session to cloud DB

Usage:
    python live_webcam.py --exercise Ex1
    python live_webcam.py --exercise Ex1 --camera 0
    python live_webcam.py --exercise Ex1 --no-movenet   # synthetic skeleton test
    python live_webcam.py --exercise Ex1 --no-tts       # disable audio
"""

import os
import sys
import argparse
import time
import threading
import collections
import numpy as np
import cv2

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.join(ROOT, 'subteam1_edge')
sys.path.insert(0, ROOT)
sys.path.insert(0, EDGE)

# ── Edge module imports ──────────────────────────────────────────────────────
from normalization import SkeletonNormalizer
from reference_model import ReferenceModel
from comparison import MovementComparator
from confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD
from imputation import JointImputer
from compensation import CompensationDetector
from fluidity import FluididtyAnalyzer

try:
    from movenet import MoveNetPoseEstimator, SKELETON_CONNECTIONS, TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

# ── TTS setup (optional) ─────────────────────────────────────────────────────
TTS_AVAILABLE = False
_tts_engine = None
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 155)   # slightly slower for clarity
    _tts_engine.setProperty('volume', 0.9)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ── Cloud import (optional) ──────────────────────────────────────────────────
try:
    from subteam2_cloud.database import RehabDatabase
    from subteam2_cloud.pubsub import MessageBroker, EdgePublisher, CloudSubscriber
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_NAME   = "RehabAI — Live Exercise Feedback"
TARGET_FRAMES = 50          # frames resampled to before classification

# Action Segmentation thresholds
# NOTE: MoveNet normalised coords are in [0,1]. At 30fps breathing/balance
# produces ~0.008-0.015 of baseline micro-movement, so thresholds need to
# be set comfortably above that noise floor.
START_VEL_THRESHOLD = 0.040  # mean joint velocity to trigger RECORDING 
STOP_VEL_THRESHOLD  = 0.025  # mean joint velocity to end RECORDING 
STOP_PATIENCE       = 10     # consecutive low-velocity frames before ANALYSING (lowered so it stops faster)
MAX_RECORD_FRAMES   = 250    # safety cap (~8 s at 30 fps)
COOLDOWN_SECS       = 2.5    # seconds to show result before returning to IDLE

SHOW_CONF_THRESHOLD = 0.3   # skip drawing joints below this confidence

# Colours (BGR)
C_GREEN  = (34, 197, 120)
C_RED    = (60, 70, 245)
C_YELLOW = (0, 210, 255)
C_BLUE   = (220, 130, 30)
C_WHITE  = (255, 255, 255)
C_DARK   = (20, 20, 30)
C_GREY   = (130, 130, 140)
C_ACCENT = (220, 100, 60)

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX

# State Machine states
STATE_IDLE      = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_ANALYSING = "ANALYSING"
STATE_COOLDOWN  = "COOLDOWN"


# ── Helpers ──────────────────────────────────────────────────────────────────

def resample_seq(seq: np.ndarray, target: int) -> np.ndarray:
    """Linear resample (frames, 17, 2) -> (target, 17, 2)."""
    n = len(seq)
    if n == target:
        return seq
    idx = np.linspace(0, n - 1, target)
    out = np.zeros((target, seq.shape[1], seq.shape[2]))
    for i, x in enumerate(idx):
        lo = int(np.floor(x))
        hi = min(lo + 1, n - 1)
        w = x - lo
        out[i] = (1 - w) * seq[lo] + w * seq[hi]
    return out


def compute_velocity(prev_kp: np.ndarray, curr_kp: np.ndarray) -> float:
    """Mean inter-frame Euclidean distance across primary joints (xy only)."""
    # Use wrists (9,10), elbows (7,8), knees (13,14), ankles (15,16) = dynamic joints
    joints = [7, 8, 9, 10, 13, 14, 15, 16]
    diffs = curr_kp[joints, :2] - prev_kp[joints, :2]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def speak_async(text: str):
    """Speak TTS in a background thread so it won't block the OpenCV loop."""
    if not TTS_AVAILABLE or _tts_engine is None:
        return
    def _speak():
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.7):
    """Semi-transparent rounded rectangle overlay."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bar(img, x, y, w, h, value, color_fg, color_bg=(50, 50, 60), label=""):
    """Horizontal progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, -1)
    filled = int(w * max(0, min(1, value)))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color_fg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_GREY, 1)
    if label:
        cv2.putText(img, label, (x, y - 6), FONT_MONO, 0.45, C_WHITE, 1, cv2.LINE_AA)


def put_text_shadow(img, text, pos, font, scale, color, thickness=1):
    """Draw text with a dark shadow."""
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), font, scale, C_DARK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, color, conf_thr=SHOW_CONF_THRESHOLD):
    """Draw keypoints + connections on frame. keypoints=(17,3) (x,y,conf) norm."""
    h, w = frame.shape[:2]
    for px, py, pc in keypoints:
        if pc >= conf_thr:
            cv2.circle(frame, (int(px * w), int(py * h)), 5, color, -1)
            cv2.circle(frame, (int(px * w), int(py * h)), 7, C_WHITE, 1)
    for i, j in SKELETON_CONNECTIONS:
        if keypoints[i, 2] >= conf_thr and keypoints[j, 2] >= conf_thr:
            x1, y1 = int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)
            x2, y2 = int(keypoints[j, 0] * w), int(keypoints[j, 1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_pulse_ring(img, cx, cy, r, state_name, thickness=6):
    """Draw a ring whose color and fill reflect the current state."""
    if state_name == STATE_IDLE:
        cv2.circle(img, (cx, cy), r, (60, 60, 70), thickness)
        cv2.circle(img, (cx, cy), r, C_GREY, thickness)
    elif state_name == STATE_RECORDING:
        cv2.circle(img, (cx, cy), r, C_RED, thickness)
        # Pulsing inner dot
        cv2.circle(img, (cx, cy), r // 3, C_RED, -1)
    elif state_name == STATE_ANALYSING:
        cv2.circle(img, (cx, cy), r, C_YELLOW, thickness)
        cv2.circle(img, (cx, cy), r // 3, C_YELLOW, -1)
    elif state_name == STATE_COOLDOWN:
        cv2.circle(img, (cx, cy), r, C_GREEN, thickness)


# ── Analysis state ───────────────────────────────────────────────────────────

class AnalysisState:
    def __init__(self):
        self.prediction      = None   # None | 0 | 1
        self.rmse            = 0.0
        self.confidence      = 0.0
        self.fluidity        = 0.0
        self.fluidity_interp = ""
        self.comp_found      = False
        self.comp_types      = []
        self.comp_severity   = 0.0
        self.method          = ""
        self.session_count   = 0
        self.rep_count       = 0
        self.machine_state   = STATE_IDLE   # current SM state

    @property
    def status(self):
        if self.prediction is None:
            return "collecting"
        return "correct" if self.prediction == 1 else "incorrect"


# ── Main overlay renderer ─────────────────────────────────────────────────────

def draw_overlay(frame, state: AnalysisState, exercise_id: str,
                 rec_frames: int, use_tts: bool = True):
    h, w = frame.shape[:2]
    sm = state.machine_state

    # ── Left sidebar ─────────────────────────────────────────────────────────
    sidebar_w = 280
    draw_rounded_rect(frame, 0, 0, sidebar_w, h, 0, (20, 22, 35), alpha=0.82)

    # ── TOP bar ───────────────────────────────────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 52, 0, (20, 22, 35), alpha=0.88)
    put_text_shadow(frame, "RehabAI", (10, 36), FONT, 1.0, C_ACCENT, 2)
    put_text_shadow(frame, f"Live Feedback   [{exercise_id}]",
                    (130, 36), FONT_MONO, 0.6, C_WHITE, 1)
    sess_txt = f"Reps: {state.rep_count}   Sessions: {state.session_count}"
    tsz = cv2.getTextSize(sess_txt, FONT_MONO, 0.5, 1)[0]
    put_text_shadow(frame, sess_txt, (w - tsz[0] - 10, 36), FONT_MONO, 0.5, C_GREY)

    # ── VERDICT / STATE banner ────────────────────────────────────────────────
    if sm == STATE_IDLE:
        verdict_color = C_GREY
        verdict_text  = "IDLE — Waiting for Movement"
        verdict_icon  = ""
    elif sm == STATE_RECORDING:
        verdict_color = C_RED
        verdict_text  = f"RECORDING... ({rec_frames} frames)"
        verdict_icon  = "●"
    elif sm == STATE_ANALYSING:
        verdict_color = C_YELLOW
        verdict_text  = "ANALYSING..."
        verdict_icon  = "⚙"
    elif sm == STATE_COOLDOWN:
        if state.prediction == 1:
            verdict_color = C_GREEN
            verdict_text  = "CORRECT"
            verdict_icon  = "[OK]"
        elif state.prediction == 0:
            verdict_color = C_RED
            verdict_text  = "INCORRECT"
            verdict_icon  = "[!]"
        else:
            verdict_color = C_GREY
            verdict_text  = "RESULT"
            verdict_icon  = ""
    else:
        verdict_color = C_YELLOW
        verdict_text  = sm
        verdict_icon  = ""

    bx1, by1, bx2, by2 = sidebar_w + 10, 62, w - 10, 140
    draw_rounded_rect(frame, bx1, by1, bx2, by2, 10, verdict_color, alpha=0.88)
    combined = (verdict_icon + " " + verdict_text).strip()
    tsz = cv2.getTextSize(combined, FONT, 1.1, 3)[0]
    tx = bx1 + (bx2 - bx1 - tsz[0]) // 2
    ty = by1 + (by2 - by1 + tsz[1]) // 2
    cv2.putText(frame, combined, (tx, ty), FONT, 1.1, C_WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, combined, (tx, ty), FONT, 1.1, C_DARK, 1, cv2.LINE_AA)

    # ── Sidebar metrics ───────────────────────────────────────────────────────
    sy = 70

    # State ring
    draw_pulse_ring(frame, 38, sy + 38, 28, sm)
    sm_label = {"IDLE": "Idle", "RECORDING": "Rec", "ANALYSING": "AI", "COOLDOWN": "Done"}.get(sm, sm)
    tsz2 = cv2.getTextSize(sm_label, FONT_MONO, 0.38, 1)[0]
    cv2.putText(frame, sm_label, (38 - tsz2[0]//2, sy + 38 + tsz2[1]//2),
                FONT_MONO, 0.38, C_WHITE, 1, cv2.LINE_AA)
    put_text_shadow(frame, "State", (74, sy + 26), FONT_MONO, 0.44, C_GREY)
    if sm == STATE_RECORDING:
        put_text_shadow(frame, f"{rec_frames} frames captured",
                        (74, sy + 50), FONT_MONO, 0.40, C_WHITE)
    sy += 88

    # -- RMSE --
    put_text_shadow(frame, "RMSE (lower=better)", (8, sy), FONT_MONO, 0.40, C_GREY)
    sy += 16
    rmse_color = C_GREEN if state.rmse < 0.30 else (C_YELLOW if state.rmse < 0.45 else C_RED)
    draw_bar(frame, 8, sy, sidebar_w - 18, 14,
             min(state.rmse / 0.6, 1.0), rmse_color)
    put_text_shadow(frame, f"{state.rmse:.3f}", (sidebar_w - 54, sy + 12),
                    FONT_MONO, 0.44, C_WHITE)
    sy += 30

    # -- Confidence --
    put_text_shadow(frame, "Confidence", (8, sy), FONT_MONO, 0.40, C_GREY)
    sy += 16
    conf_color = C_GREEN if state.confidence >= 0.7 else (C_YELLOW if state.confidence >= 0.5 else C_RED)
    draw_bar(frame, 8, sy, sidebar_w - 18, 14, state.confidence, conf_color)
    put_text_shadow(frame, f"{state.confidence*100:.0f}%", (sidebar_w - 46, sy + 12),
                    FONT_MONO, 0.44, C_WHITE)
    sy += 30

    # -- Fluidity --
    put_text_shadow(frame, "Fluidity", (8, sy), FONT_MONO, 0.40, C_GREY)
    sy += 16
    flu_color = C_GREEN if state.fluidity >= 0.7 else (C_YELLOW if state.fluidity >= 0.4 else C_RED)
    draw_bar(frame, 8, sy, sidebar_w - 18, 14, state.fluidity, flu_color)
    put_text_shadow(frame, f"{state.fluidity*100:.0f}%", (sidebar_w - 46, sy + 12),
                    FONT_MONO, 0.44, C_WHITE)
    sy += 20
    interp_short = state.fluidity_interp[:28] if state.fluidity_interp else ""
    put_text_shadow(frame, interp_short, (8, sy + 12), FONT_MONO, 0.36, C_GREY)
    sy += 28

    # -- Compensation --
    sy += 6
    put_text_shadow(frame, "Compensation", (8, sy), FONT_MONO, 0.40, C_GREY)
    sy += 20
    if state.comp_found:
        draw_rounded_rect(frame, 6, sy - 4, sidebar_w - 8, sy + 14 * len(state.comp_types) + 10,
                          4, (30, 40, 90), alpha=0.75)
        for ctype in state.comp_types:
            short = {'trunk_lean': 'Trunk Lean!',
                     'hip_shift': 'Hip Shift!',
                     'shoulder_shrug': 'Shoulder Shrug!'}.get(ctype, ctype)
            put_text_shadow(frame, f"  ! {short}", (8, sy + 12), FONT_MONO, 0.42, C_YELLOW)
            sy += 16
        severity_pct = int(state.comp_severity * 100)
        put_text_shadow(frame, f"  Severity: {severity_pct}%", (8, sy + 12),
                        FONT_MONO, 0.38, C_GREY)
        sy += 18
    else:
        put_text_shadow(frame, "  None detected", (8, sy + 12), FONT_MONO, 0.42, C_GREEN)
        sy += 20

    # -- Method label --
    if state.method:
        sy = max(sy + 8, h - 100)
        put_text_shadow(frame, f"Method: {state.method}", (8, sy), FONT_MONO, 0.36, C_GREY)

    # ── BOTTOM instructions bar ───────────────────────────────────────────────
    draw_rounded_rect(frame, 0, h - 36, w, h, 0, (20, 22, 35), alpha=0.88)
    hints = "Q - Quit    S - Save session to Cloud"
    th = cv2.getTextSize(hints, FONT_MONO, 0.42, 1)[0]
    cv2.putText(frame, hints, (w // 2 - th[0] // 2, h - 10),
                FONT_MONO, 0.42, C_GREY, 1, cv2.LINE_AA)

    return frame


# ── Main live loop ────────────────────────────────────────────────────────────

def run_live(
    exercise_id: str = "Ex1",
    dataset_path: str = "dataset",
    camera_idx: int = 0,
    threshold: float = 0.30,
    use_movenet: bool = True,
    patient_id: str = "webcam_patient",
    db_path: str = "rehab_data",
    use_tts: bool = True,
):
    """
    Main live webcam feedback loop with Kinematic State Machine.

    Parameters
    ----------
    exercise_id   : str   — e.g. "Ex1"
    dataset_path  : str   — path to REHAB24-6 dataset
    camera_idx    : int   — OpenCV camera index
    threshold     : float — RMSE fallback threshold if no CNN model found
    use_movenet   : bool  — use live MoveNet (requires TF + webcam)
    patient_id    : str   — for cloud session upload
    db_path       : str   — cloud database path
    use_tts       : bool  — enable audio feedback
    """

    print("\n" + "=" * 60)
    print("  RehabAI Live Webcam Mode  (Action Segmentation)")
    print(f"  Exercise : {exercise_id}")
    print(f"  Camera   : {camera_idx}")
    print(f"  TF/MoveNet: {TF_AVAILABLE}")
    print(f"  TTS Audio : {TTS_AVAILABLE and use_tts}")
    print("=" * 60)

    # ── 1. Build reference model ─────────────────────────────────────────────
    print("\nBuilding reference model from dataset...")
    ref_model = ReferenceModel(dataset_path)
    try:
        stats = ref_model.build_from_samples(exercise_id, camera="c17", fps=30,
                                              target_frames=TARGET_FRAMES)
        print(f"  Reference built from {stats['num_samples']} correct samples")
        print(f"  Reference shape: {stats['mean_shape']}")
    except Exception as e:
        print(f"  [!] Reference model failed: {e}")
        print("  Make sure --dataset points to the REHAB24-6 dataset folder.")
        return

    comparator    = MovementComparator(ref_model.reference_mean, ref_model.reference_std)
    normaliser    = SkeletonNormalizer()
    imputer       = JointImputer(confidence_threshold=0.3)
    comp_detector = CompensationDetector(severity_threshold=0.3)
    fluidity      = FluididtyAnalyzer(target_frames=TARGET_FRAMES)
    conf_scorer   = ConfidenceScorer()

    # ── 2. MoveNet setup ─────────────────────────────────────────────────────
    estimator = None
    if use_movenet and TF_AVAILABLE:
        print("\nLoading MoveNet Lightning (Fast)...")
        try:
            estimator = MoveNetPoseEstimator(use_lightning=True)
            print("  MoveNet ready.")
        except Exception as e:
            print(f"  MoveNet load failed: {e}")

    if estimator is None:
        print("  Running WITHOUT MoveNet (synthetic random skeleton for UI test).")

    # ── 3. Load Unified CNN Model (auto-detect, prefer over math threshold) ───
    cnn_model = None
    model_path = os.path.join(ROOT, "models", "cnn_classifier_all_exercises.keras")
    if TF_AVAILABLE and os.path.exists(model_path):
        from tensorflow import keras
        print(f"\nLoading Unified Neural Network from {model_path}...")
        try:
            cnn_model = keras.models.load_model(model_path)
            print("  CNN Model (All 6 Exercises) successfully loaded!")
        except Exception as e:
            print(f"  [!] Failed to load CNN: {e} — falling back to RMSE threshold.")
    elif not os.path.exists(model_path):
        print(f"\n[INFO] No saved model at {model_path}.")
        print("  Run `python train_all.py` to train and save the global model.")
        print("  Using RMSE Math Threshold as fallback.")

    # ── 4. Cloud setup (optional) ─────────────────────────────────────────────
    broker   = None
    edge_pub = None
    db       = None
    if CLOUD_AVAILABLE:
        db        = RehabDatabase(db_path=db_path)
        broker    = MessageBroker(mode="local")
        cloud_sub = CloudSubscriber(broker, database=db)
        edge_pub  = EdgePublisher(broker, edge_id="edge_webcam")
        broker.start()
        print(f"  Cloud DB ready ({db.total_sessions()} existing sessions)")

    # ── 5. Open webcam ───────────────────────────────────────────────────────
    synthetic_mode = (estimator is None)
    cap = None

    if not synthetic_mode:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"\n[ERROR] Cannot open camera index {camera_idx}.")
            print("  Falling back to synthetic mode for UI test.")
            cap = None
            synthetic_mode = True

    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n  Camera opened: {actual_w}x{actual_h}")
    else:
        actual_w, actual_h = 1280, 720
        print(f"\n  Running in synthetic mode: {actual_w}x{actual_h} canvas")

    print("  Press 'q' to quit, 's' to save session.\n")

    # ── 6. Initialise State Machine variables ─────────────────────────────────
    state           = AnalysisState()
    state.machine_state = STATE_IDLE

    # Rolling small buffer for imputation context (always kept, 5 frames)
    kp_buffer_impute = collections.deque(maxlen=5)

    # Dynamic recording buffer (grows during RECORDING)
    record_buffer: list = []      # list of (17,2) normalised frames

    # Velocity tracking
    prev_norm_kp = None           # previous frame's (17,3) imputed keypoints
    stop_counter = 0              # consecutive low-velocity frames
    cooldown_start = 0.0          # timestamp when COOLDOWN began

    # For saving sessions
    last_seq_resampled = None

    frame_idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_w, 1440), min(actual_h, 810))

    if use_tts and TTS_AVAILABLE:
        speak_async(f"RehabAI ready. Exercise {exercise_id}. Begin when ready.")

    print(f"  Live loop started. State: {STATE_IDLE}\n")

    # ── 7. Main loop ─────────────────────────────────────────────────────────
    while True:
        # ── 7a. Grab frame ────────────────────────────────────────────────────
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
        else:
            frame = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
            frame[:] = (28, 28, 38)
            time.sleep(0.033)

        # ── 7b. Pose estimation ───────────────────────────────────────────────
        if estimator is not None:
            try:
                raw_kp = estimator.extract_keypoints(frame)   # (17,3)
            except Exception:
                raw_kp = np.zeros((17, 3))
        else:
            raw_kp = _synthetic_keypoints(frame_idx)

        # ── 7c. Imputation (temporal context) ─────────────────────────────────
        kp_buffer_impute.append(raw_kp.copy())
        if len(kp_buffer_impute) >= 3:
            recent = np.array(list(kp_buffer_impute))
            imputed_seq, _ = imputer.impute_sequence(recent)
            cur_kp_imputed = imputed_seq[-1]
        else:
            cur_kp_imputed = raw_kp.copy()

        # ── 7d. Normalise ──────────────────────────────────────────────────────
        kp_3ch = cur_kp_imputed[np.newaxis, :, :]
        try:
            norm = normaliser.normalize_sequence(kp_3ch)
            norm_xy = norm[0, :, :2]     # (17, 2)
        except Exception:
            norm_xy = cur_kp_imputed[:, :2]

        # ── 7e. Compute velocity (for state machine) ───────────────────────────
        if prev_norm_kp is not None:
            velocity = compute_velocity(prev_norm_kp, cur_kp_imputed)
        else:
            velocity = 0.0
        prev_norm_kp = cur_kp_imputed.copy()

        # ── 7f. State Machine transitions ────────────────────────────────────
        sm = state.machine_state

        if sm == STATE_IDLE:
            if velocity > START_VEL_THRESHOLD:
                state.machine_state = STATE_RECORDING
                record_buffer = [norm_xy.copy()]
                stop_counter = 0
                print(f"  [SM] IDLE -> RECORDING  (vel={velocity:.4f})")
                if use_tts and TTS_AVAILABLE:
                    speak_async("Recording.")

        elif sm == STATE_RECORDING:
            record_buffer.append(norm_xy.copy())

            if velocity < STOP_VEL_THRESHOLD:
                stop_counter += 1
            else:
                stop_counter = 0

            # Transition to ANALYSING if still or too long
            if stop_counter >= STOP_PATIENCE or len(record_buffer) >= MAX_RECORD_FRAMES:
                state.machine_state = STATE_ANALYSING
                print(f"  [SM] RECORDING -> ANALYSING  "
                      f"({len(record_buffer)} frames, vel={velocity:.4f})")

        elif sm == STATE_ANALYSING:
            # Only do AI inference if we have enough frames
            if len(record_buffer) >= 10:
                seq = np.array(record_buffer)              # (N, 17, 2)
                seq_resampled = resample_seq(seq, TARGET_FRAMES)
                last_seq_resampled = seq_resampled

                try:
                    # Comparison (RMSE — always run for metrics/compensation)
                    metrics    = comparator.compare_sequence(seq_resampled)
                    rmse       = float(metrics['overall_rmse'])
                    confidence = float(conf_scorer.score_from_rmse(rmse))

                    # Classification: prefer CNN, fallback to threshold
                    if cnn_model is not None:
                        proba = float(
                            cnn_model.predict(seq_resampled[np.newaxis, ...], verbose=0)[0, 0]
                        )
                        prediction = 1 if proba >= 0.5 else 0
                        method = f"CNN (p={proba:.2f})"
                    else:
                        prediction = 1 if rmse <= threshold else 0
                        method = f"Threshold ({threshold:.2f})"

                    # Compensation & fluidity
                    comp = comp_detector.detect(
                        seq_resampled, exercise_id, ref_model.reference_mean)
                    flu  = fluidity.analyze(seq_resampled)

                    # Update analysis state
                    state.prediction      = prediction
                    state.rmse            = rmse
                    state.confidence      = confidence
                    state.fluidity        = flu['overall_fluidity']
                    state.fluidity_interp = flu['interpretation']
                    state.comp_found      = comp['compensation_found']
                    state.comp_types      = comp['types']
                    state.comp_severity   = comp['severity']
                    state.method          = method
                    state.rep_count      += 1

                    # Build TTS message
                    if use_tts and TTS_AVAILABLE:
                        verdict_word = "Correct" if prediction == 1 else "Incorrect"
                        tts_parts = [f"Repetition {state.rep_count}. {verdict_word}."]
                        if comp['compensation_found']:
                            types_str = " and ".join(
                                t.replace("_", " ") for t in comp['types']
                            )
                            tts_parts.append(f"Warning: {types_str} detected.")
                        if flu['overall_fluidity'] < 0.5:
                            tts_parts.append("Try to move more smoothly.")
                        speak_async(" ".join(tts_parts))

                    print(f"  [SM] ANALYSING complete -> "
                          f"{'CORRECT' if prediction == 1 else 'INCORRECT'} "
                          f"RMSE={rmse:.3f}  method={method}")

                except Exception as ex:
                    print(f"  [SM] Analysis error: {ex}")

            # Transition to COOLDOWN regardless
            state.machine_state = STATE_COOLDOWN
            cooldown_start = time.time()
            record_buffer = []

        elif sm == STATE_COOLDOWN:
            if time.time() - cooldown_start >= COOLDOWN_SECS:
                state.machine_state = STATE_IDLE
                stop_counter = 0
                print(f"  [SM] COOLDOWN -> IDLE")

        # ── 7g. Skeleton colour based on state / last verdict ─────────────────
        if sm == STATE_RECORDING:
            skel_color = C_YELLOW
        elif sm == STATE_COOLDOWN and state.prediction == 1:
            skel_color = C_GREEN
        elif sm == STATE_COOLDOWN and state.prediction == 0:
            skel_color = C_RED
        elif sm == STATE_IDLE:
            skel_color = C_GREY
        else:
            skel_color = C_YELLOW

        draw_skeleton(frame, cur_kp_imputed, skel_color)

        # ── 7h. Draw overlay ──────────────────────────────────────────────────
        draw_overlay(frame, state, exercise_id,
                     len(record_buffer), use_tts)

        cv2.imshow(WINDOW_NAME, frame)

        # ── 7i. Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  Quitting...")
            break

        elif key == ord('s'):
            if state.prediction is not None and last_seq_resampled is not None:
                session_data = {
                    'patient_id':              patient_id,
                    'exercise_id':             exercise_id,
                    'correctness':             state.prediction,
                    'confidence':              round(state.confidence, 4),
                    'rmse':                    round(state.rmse, 4),
                    'compensation_found':      state.comp_found,
                    'compensation_types':      state.comp_types,
                    'compensation_severity':   round(state.comp_severity, 3),
                    'fluidity_score':          state.fluidity,
                    'fluidity_interpretation': state.fluidity_interp,
                    'prediction_method':       state.method,
                    'source':                  'webcam',
                }
                if CLOUD_AVAILABLE and edge_pub is not None:
                    edge_pub.publish_session(session_data)
                    if broker:
                        broker.flush(timeout=0.5)
                    state.session_count += 1
                    total = db.total_sessions() if db else '?'
                    print(f"  [SAVED] Session #{state.session_count} -> DB total: {total}")
                elif db is not None:
                    sid = db.save_session(session_data)
                    state.session_count += 1
                    print(f"  [SAVED] session_id={sid[:12]}...")
                else:
                    print("  [!] Cloud not available — install flask/tinydb.")
            else:
                print("  [!] No rep completed yet — complete an exercise first.")

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if broker:
        broker.stop()
    print(f"\n  Reps completed: {state.rep_count}")
    print(f"  Sessions saved: {state.session_count}")
    print("  Live webcam mode closed.")


# ── Synthetic keypoints (for UI testing without MoveNet/webcam) ───────────────

def _synthetic_keypoints(t: int) -> np.ndarray:
    """Generate a plausible standing pose with slight oscillation."""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.95

    base = np.array([
        [0.50, 0.13],  # nose
        [0.48, 0.11],  # left_eye
        [0.52, 0.11],  # right_eye
        [0.46, 0.12],  # left_ear
        [0.54, 0.12],  # right_ear
        [0.43, 0.25],  # left_shoulder
        [0.57, 0.25],  # right_shoulder
        [0.38, 0.40],  # left_elbow
        [0.62, 0.40],  # right_elbow
        [0.35, 0.55],  # left_wrist
        [0.65, 0.55],  # right_wrist
        [0.44, 0.58],  # left_hip
        [0.56, 0.58],  # right_hip
        [0.44, 0.75],  # left_knee
        [0.56, 0.75],  # right_knee
        [0.44, 0.92],  # left_ankle
        [0.56, 0.92],  # right_ankle
    ])

    # Simulate a slow arm raise oscillation on left wrist (index 9)
    osc = 0.12 * np.sin(t / 20.0)
    base[9, 1] -= osc   # left wrist moves up/down

    noise = np.random.normal(0, 0.003, base.shape)
    kp[:, :2] = np.clip(base + noise, 0.05, 0.95)
    return kp


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RehabAI Live Webcam — Real-Time Exercise Feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_webcam.py --exercise Ex1
  python live_webcam.py --exercise Ex2 --camera 1
  python live_webcam.py --exercise Ex1 --threshold 0.28
  python live_webcam.py --exercise Ex1 --no-movenet   # UI test without TF/webcam
  python live_webcam.py --exercise Ex1 --no-tts       # disable audio feedback
  python live_webcam.py --exercise Ex1 --patient patient_001 --db rehab_data
        """,
    )
    parser.add_argument("--exercise",   "-e", default="Ex1",
                        choices=["Ex1","Ex2","Ex3","Ex4","Ex5","Ex6"],
                        help="Exercise to perform (default: Ex1)")
    parser.add_argument("--dataset",    "-d", default="dataset",
                        help="Path to REHAB24-6 dataset (default: dataset)")
    parser.add_argument("--camera",     "-c", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--threshold",  "-t", type=float, default=0.30,
                        help="RMSE fallback threshold (default: 0.30)")
    parser.add_argument("--no-movenet", action="store_true",
                        help="Disable MoveNet — use synthetic skeleton (UI test)")
    parser.add_argument("--no-tts",     action="store_true",
                        help="Disable audio feedback")
    parser.add_argument("--patient",    default="webcam_patient",
                        help="Patient ID for cloud session upload")
    parser.add_argument("--db",         default="rehab_data",
                        help="Cloud database path (default: rehab_data)")

    args = parser.parse_args()

    run_live(
        exercise_id  = args.exercise,
        dataset_path = args.dataset,
        camera_idx   = args.camera,
        threshold    = args.threshold,
        use_movenet  = not args.no_movenet,
        patient_id   = args.patient,
        db_path      = args.db,
        use_tts      = not args.no_tts,
    )


if __name__ == "__main__":
    main()
