"""
Live Webcam Mode — Real-Time Exercise Feedback for Patients
============================================================
Open the webcam, detect pose with MoveNet, classify the exercise
in a rolling window of N frames, and display clear patient-facing
feedback on the video feed.

Feedback shown to the patient:
  - Skeleton overlay (GREEN = correct, RED = incorrect, YELLOW = collecting)
  - Large verdict banner: CORRECT / INCORRECT / COLLECTING DATA
  - RMSE similarity bar
  - Confidence bar
  - Fluidity score
  - Compensation warnings
  - Frame-count progress ring
  - Press 'q' to quit | 's' to save session to cloud DB

Usage:
    cd subteam1_edge
    python ../live_webcam.py --exercise Ex1
    python ../live_webcam.py --exercise Ex1 --camera 0 --window 50
    python ../live_webcam.py --exercise Ex1 --no-movenet   # synthetic skeleton test
"""

import os
import sys
import argparse
import time
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

# ── Cloud import (optional) ──────────────────────────────────────────────────
try:
    from subteam2_cloud.database import RehabDatabase
    from subteam2_cloud.pubsub import MessageBroker, EdgePublisher, CloudSubscriber
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_NAME = "RehabAI — Live Exercise Feedback"
TARGET_FRAMES = 50          # frames per analysis window
ANALYSE_EVERY = 10          # run classifier every N new frames (after window full)
SHOW_CONF_THRESHOLD = 0.3   # skip drawing joints below this confidence

# Colours (BGR)
C_GREEN  = (34, 197, 120)
C_RED    = (60, 70, 245)
C_YELLOW = (0, 210, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (20, 20, 30)
C_GREY   = (130, 130, 140)
C_ACCENT = (220, 100, 60)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


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


def draw_ring(img, cx, cy, r, value, color, thickness=6):
    """Draw a progress ring (0-1 fill)."""
    cv2.circle(img, (cx, cy), r, (60, 60, 70), thickness)
    angle = int(360 * max(0, min(1, value)))
    axes = (r, r)
    cv2.ellipse(img, (cx, cy), axes, -90, 0, angle, color, thickness, cv2.LINE_AA)


# ── Analysis state (cached between frames) ───────────────────────────────────

class AnalysisState:
    def __init__(self):
        self.prediction    = None   # None | 0 | 1
        self.rmse          = 0.0
        self.confidence    = 0.0
        self.fluidity      = 0.0
        self.fluidity_interp = ""
        self.comp_found    = False
        self.comp_types    = []
        self.comp_severity = 0.0
        self.method        = ""
        self.frames_in_buf = 0
        self.last_analysed = 0.0    # timestamp
        self.session_count = 0

    @property
    def status(self):
        if self.prediction is None:
            return "collecting"
        return "correct" if self.prediction == 1 else "incorrect"


# ── Main overlay drawing ─────────────────────────────────────────────────────

def draw_overlay(frame, state: AnalysisState, exercise_id: str,
                 buf_len: int, window: int):
    h, w = frame.shape[:2]
    fill = buf_len / window  # 0→1

    # ── Left sidebar background ──────────────────────────────────────────────
    sidebar_w = 280
    draw_rounded_rect(frame, 0, 0, sidebar_w, h, 0, (20, 22, 35), alpha=0.82)

    # ── TOP bar ──────────────────────────────────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 52, 0, (20, 22, 35), alpha=0.88)

    put_text_shadow(frame, "RehabAI", (10, 36), FONT, 1.0, C_ACCENT, 2)
    put_text_shadow(frame, f"Live Feedback   [{exercise_id}]",
                    (130, 36), FONT_MONO, 0.6, C_WHITE, 1)

    # Session counter top-right
    sess_txt = f"Sessions: {state.session_count}"
    tsz = cv2.getTextSize(sess_txt, FONT_MONO, 0.5, 1)[0]
    put_text_shadow(frame, sess_txt, (w - tsz[0] - 10, 36), FONT_MONO, 0.5, C_GREY)

    # ── VERDICT banner ────────────────────────────────────────────────────────
    st = state.status
    if st == "collecting":
        verdict_color = C_YELLOW
        verdict_text  = "COLLECTING DATA..."
        verdict_icon  = ""
    elif st == "correct":
        verdict_color = C_GREEN
        verdict_text  = "CORRECT"
        verdict_icon  = "[OK]"
    else:
        verdict_color = C_RED
        verdict_text  = "INCORRECT"
        verdict_icon  = "[!]"

    # Large banner box
    bx1, by1, bx2, by2 = sidebar_w + 10, 62, w - 10, 140
    draw_rounded_rect(frame, bx1, by1, bx2, by2, 10, verdict_color, alpha=0.88)

    tsz = cv2.getTextSize(verdict_icon + " " + verdict_text, FONT, 1.3, 3)[0]
    tx = bx1 + (bx2 - bx1 - tsz[0]) // 2
    ty = by1 + (by2 - by1 + tsz[1]) // 2
    cv2.putText(frame, verdict_icon + " " + verdict_text,
                (tx, ty), FONT, 1.3, C_WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, verdict_icon + " " + verdict_text,
                (tx, ty), FONT, 1.3, C_DARK, 1, cv2.LINE_AA)

    # ── Sidebar metrics ──────────────────────────────────────────────────────
    sy = 70   # current Y position in sidebar

    # -- Buffer fill ring --
    draw_ring(frame, 38, sy + 38, 28, fill, verdict_color)
    pct_txt = f"{int(fill*100)}%"
    tsz2 = cv2.getTextSize(pct_txt, FONT_MONO, 0.4, 1)[0]
    cv2.putText(frame, pct_txt, (38 - tsz2[0]//2, sy + 38 + tsz2[1]//2),
                FONT_MONO, 0.4, C_WHITE, 1, cv2.LINE_AA)
    put_text_shadow(frame, "Buffer", (74, sy + 26), FONT_MONO, 0.44, C_GREY)
    put_text_shadow(frame, f"{buf_len}/{window} frames",
                    (74, sy + 50), FONT_MONO, 0.44, C_WHITE)
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
    # fluidity label
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
        sy += 16

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
    window: int = TARGET_FRAMES,
    threshold: float = 0.30,
    use_movenet: bool = True,
    patient_id: str = "webcam_patient",
    db_path: str = "rehab_data",
):
    """
    Main live webcam feedback loop.

    Parameters
    ----------
    exercise_id   : str   — e.g. "Ex1"
    dataset_path  : str   — path to REHAB24-6 dataset (needed for reference model)
    camera_idx    : int   — OpenCV camera index (0 = default webcam)
    window        : int   — number of frames to accumulate before classifying
    threshold     : float — RMSE threshold for correct/incorrect
    use_movenet   : bool  — use live MoveNet (requires TF + webcam)
    patient_id    : str   — for cloud session upload
    db_path       : str   — cloud database path
    """

    print("\n" + "=" * 60)
    print("  RehabAI Live Webcam Mode")
    print(f"  Exercise : {exercise_id}")
    print(f"  Camera   : {camera_idx}")
    print(f"  Window   : {window} frames")
    print(f"  TF/MoveNet: {TF_AVAILABLE}")
    print("=" * 60)

    # ── 1. Build reference model ─────────────────────────────────────────────
    print("\nBuilding reference model from dataset...")
    ref_model = ReferenceModel(dataset_path)
    try:
        stats = ref_model.build_from_samples(exercise_id, camera="c17", fps=30,
                                              target_frames=window)
        print(f"  Reference built from {stats['num_samples']} correct samples")
        print(f"  Reference shape: {stats['mean_shape']}")
    except Exception as e:
        print(f"  [!] Reference model failed: {e}")
        print("  Make sure --dataset points to the REHAB24-6 dataset folder.")
        return

    comparator     = MovementComparator(ref_model.reference_mean, ref_model.reference_std)
    normaliser     = SkeletonNormalizer()
    imputer        = JointImputer(confidence_threshold=0.3)
    comp_detector  = CompensationDetector(severity_threshold=0.3)
    fluidity       = FluididtyAnalyzer(target_frames=window)
    conf_scorer    = ConfidenceScorer()

    # ── 2. MoveNet setup ─────────────────────────────────────────────────────
    estimator = None
    if use_movenet and TF_AVAILABLE:
        print("\nLoading MoveNet Thunder (may take ~30s first run)...")
        try:
            estimator = MoveNetPoseEstimator()
            print("  MoveNet ready.")
        except Exception as e:
            print(f"  MoveNet load failed: {e}")
            estimator = None

    if estimator is None:
        print("  Running WITHOUT MoveNet (synthetic random skeleton for UI test).")

    # ── 3. Cloud setup (optional) ─────────────────────────────────────────────
    broker   = None
    edge_pub = None
    db       = None
    if CLOUD_AVAILABLE:
        db     = RehabDatabase(db_path=db_path)
        broker = MessageBroker(mode="local")
        cloud_sub = CloudSubscriber(broker, database=db)
        edge_pub  = EdgePublisher(broker, edge_id="edge_webcam")
        broker.start()
        print(f"  Cloud DB ready ({db.total_sessions()} existing sessions)")

    # ── 4. Open webcam ───────────────────────────────────────────────────────
    synthetic_mode = (estimator is None)
    cap = None

    if not synthetic_mode:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"\n[ERROR] Cannot open camera index {camera_idx}.")
            print("  Try a different --camera index (0, 1, 2...).")
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

    print("  Press 'q' to quit, 's' to save session.")

    # ── 5. Rolling buffer + state ─────────────────────────────────────────────
    kp_buffer  = collections.deque(maxlen=window)   # raw (17,3) per frame
    norm_buffer = collections.deque(maxlen=window)  # normalised (17,2) per frame
    state      = AnalysisState()
    frame_idx  = 0
    frames_since_last = 0
    last_kp    = np.zeros((17, 3))                  # for overlay between analysis

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_w, 1440), min(actual_h, 810))

    print("\n  Live loop started. Collecting keypoints...\n")

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed — retrying...")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)   # mirror for natural feel
        else:
            # Synthetic canvas background
            frame = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
            frame[:] = (28, 28, 38)   # dark background
            time.sleep(0.033)          # ~30 fps cap for synthetic mode

        # ── 5a. Extract keypoints ─────────────────────────────────────────
        if estimator is not None:
            try:
                raw_kp = estimator.extract_keypoints(frame)   # (17,3) x,y,conf
            except Exception:
                raw_kp = np.zeros((17, 3))
        else:
            # Synthetic: use centred positions + noise (for UI testing without TF)
            raw_kp = _synthetic_keypoints(frame_idx)

        last_kp = raw_kp

        # ── 5b. Imputation (per frame — use 1+future context window) ─────
        # Build a tiny sequence from last few frames for temporal imputation
        kp_buffer.append(raw_kp.copy())
        if len(kp_buffer) >= 3:
            recent = np.array(list(kp_buffer)[-5:])   # up to 5 frames
            imputed_seq, _ = imputer.impute_sequence(recent)
            cur_kp_imputed = imputed_seq[-1]           # latest imputed frame
        else:
            cur_kp_imputed = raw_kp.copy()

        # ── 5c. Normalise & buffer ────────────────────────────────────────
        kp_3ch = cur_kp_imputed[np.newaxis, :, :]       # (1,17,3)
        try:
            norm = normaliser.normalize_sequence(kp_3ch)  # (1,17,3)
            norm_xy = norm[0, :, :2]                      # (17,2)
        except Exception:
            norm_xy = cur_kp_imputed[:, :2]

        norm_buffer.append(norm_xy)

        # ── 5d. Draw skeleton on frame ────────────────────────────────────
        if state.status == "correct":
            skel_color = C_GREEN
        elif state.status == "incorrect":
            skel_color = C_RED
        else:
            skel_color = C_YELLOW

        draw_skeleton(frame, cur_kp_imputed, skel_color)

        # ── 5e. Run analysis when buffer is full & every N frames ─────────
        frames_since_last += 1
        if len(norm_buffer) >= window and frames_since_last >= ANALYSE_EVERY:
            frames_since_last = 0
            seq = np.array(list(norm_buffer))          # (window, 17, 2)
            seq_resampled = resample_seq(seq, window)

            try:
                # Module 4: Comparison
                metrics = comparator.compare_sequence(seq_resampled)
                rmse    = float(metrics['overall_rmse'])

                # Confidence scoring
                confidence = conf_scorer.score_from_rmse(rmse)

                # Classification (threshold-based, instant)
                prediction = 1 if rmse <= threshold else 0

                # Compensation detection
                comp = comp_detector.detect(
                    seq_resampled, exercise_id, ref_model.reference_mean)

                # Fluidity
                flu = fluidity.analyze(seq_resampled)

                # Update state
                state.prediction   = prediction
                state.rmse         = rmse
                state.confidence   = float(confidence)
                state.fluidity     = flu['overall_fluidity']
                state.fluidity_interp = flu['interpretation']
                state.comp_found   = comp['compensation_found']
                state.comp_types   = comp['types']
                state.comp_severity = comp['severity']
                state.method       = f"Threshold ({threshold:.2f})"

            except Exception as ex:
                pass   # keep last state if analysis errors

        state.frames_in_buf = len(norm_buffer)

        # ── 5f. Draw the full overlay ─────────────────────────────────────
        draw_overlay(frame, state, exercise_id, len(norm_buffer), window)

        cv2.imshow(WINDOW_NAME, frame)

        # ── 5g. Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  Quitting...")
            break

        elif key == ord('s'):
            # Save current session to cloud DB
            if state.prediction is not None:
                session_data = {
                    'patient_id': patient_id,
                    'exercise_id': exercise_id,
                    'correctness': state.prediction,
                    'confidence': round(state.confidence, 4),
                    'rmse': round(state.rmse, 4),
                    'compensation_found': state.comp_found,
                    'compensation_types': state.comp_types,
                    'compensation_severity': round(state.comp_severity, 3),
                    'fluidity_score': state.fluidity,
                    'fluidity_interpretation': state.fluidity_interp,
                    'prediction_method': 'ThresholdLive',
                    'source': 'webcam',
                }
                if CLOUD_AVAILABLE and edge_pub is not None:
                    edge_pub.publish_session(session_data)
                    if broker:
                        broker.flush(timeout=0.5)
                    state.session_count += 1
                    total = db.total_sessions() if db else '?'
                    print(f"  [SAVED] Session #{state.session_count} "
                          f"-> DB total: {total}")
                elif db is not None:
                    sid = db.save_session(session_data)
                    state.session_count += 1
                    print(f"  [SAVED] session_id={sid[:12]}...")
                else:
                    print("  [!] Cloud not available — install flask/tinydb.")
            else:
                print("  [!] Not enough data to save — wait for analysis.")

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if broker:
        broker.stop()
    print(f"\n  Sessions saved this run: {state.session_count}")
    print("  Live webcam mode closed.")


# ── Synthetic keypoints (for UI testing without MoveNet/webcam) ───────────────

def _synthetic_keypoints(t: int) -> np.ndarray:
    """Generate a plausible standing pose with slight oscillation."""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.95                       # confidence

    # Base standing pose (x, y) in [0,1]
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

    # Simulate right arm raise over time
    angle = (t % 90) / 90.0 * np.pi / 2
    base[6, 1] = 0.25 - 0.20 * np.sin(angle)   # right shoulder y
    base[8, 1] = 0.40 - 0.25 * np.sin(angle)   # right elbow y
    base[10, 1] = 0.55 - 0.30 * np.sin(angle)  # right wrist y

    noise = np.random.normal(0, 0.005, base.shape)
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
  python live_webcam.py --exercise Ex1 --window 40 --threshold 0.28
  python live_webcam.py --exercise Ex1 --no-movenet   # UI test without TF/webcam
  python live_webcam.py --exercise Ex1 --patient patient_001 --db rehab_data
        """,
    )
    parser.add_argument("--exercise",  "-e", default="Ex1",
                        choices=["Ex1","Ex2","Ex3","Ex4","Ex5","Ex6"],
                        help="Exercise to perform (default: Ex1)")
    parser.add_argument("--dataset",   "-d", default="dataset",
                        help="Path to REHAB24-6 dataset (default: dataset)")
    parser.add_argument("--camera",    "-c", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--window",    "-w", type=int, default=TARGET_FRAMES,
                        help=f"Frames per analysis window (default: {TARGET_FRAMES})")
    parser.add_argument("--threshold", "-t", type=float, default=0.30,
                        help="RMSE threshold: below=correct (default: 0.30)")
    parser.add_argument("--no-movenet", action="store_true",
                        help="Disable MoveNet — use synthetic skeleton (UI test)")
    parser.add_argument("--patient",   default="webcam_patient",
                        help="Patient ID for cloud session upload")
    parser.add_argument("--db",        default="rehab_data",
                        help="Cloud database path (default: rehab_data)")

    args = parser.parse_args()

    run_live(
        exercise_id   = args.exercise,
        dataset_path  = args.dataset,
        camera_idx    = args.camera,
        window        = args.window,
        threshold     = args.threshold,
        use_movenet   = not args.no_movenet,
        patient_id    = args.patient,
        db_path       = args.db,
    )


if __name__ == "__main__":
    main()
