"""
LSTM Fluidity Analysis Module
------------------------------
End-Semester Enhancement: Assesses the smoothness and naturalness of
rehabilitation exercise movements.

Two-tier analysis:
    1. Rule-based metrics (always available, no training needed)
       - Jerk: 3rd derivative of joint position (higher = less smooth)
       - Velocity consistency: std(velocity) across frames
       - Acceleration variance
    2. LSTM regression model (trained on dataset)
       - Input: velocity + acceleration per joint
       - Output: fluidity score in [0, 1]

Research Paper Reference (End-Semester):
    "Fluidity is assessed using temporal analysis of movement trajectories
    to distinguish smooth, coordinated movements from jerky or hesitant ones,
    providing additional clinical insight beyond binary correctness."
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ------------------------------------------------------------------
# Rule-Based Fluidity Metrics
# ------------------------------------------------------------------

def compute_velocity(position_seq: np.ndarray) -> np.ndarray:
    """
    First finite difference along the time axis.

    Parameters
    ----------
    position_seq : np.ndarray  (frames, 17, 2)

    Returns
    -------
    velocity : np.ndarray  (frames-1, 17, 2)
    """
    return np.diff(position_seq, axis=0)


def compute_acceleration(position_seq: np.ndarray) -> np.ndarray:
    """Second finite difference (frames-2, 17, 2)."""
    return np.diff(position_seq, n=2, axis=0)


def compute_jerk(position_seq: np.ndarray) -> np.ndarray:
    """Third finite difference (frames-3, 17, 2). Higher = less smooth."""
    return np.diff(position_seq, n=3, axis=0)


def compute_jerk_score(position_seq: np.ndarray) -> float:
    """
    Normalised jerk score: mean absolute jerk, inverted to [0, 1].
    Score of 1 = perfectly smooth, 0 = very jerky.
    """
    if len(position_seq) < 4:
        return 0.5  # not enough data

    jerk = compute_jerk(position_seq)          # (frames-3, 17, 2)
    mean_abs_jerk = float(np.mean(np.abs(jerk)))

    # Sigmoid-like inversion: score approaches 1 as jerk -> 0
    score = 1.0 / (1.0 + mean_abs_jerk * 10.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_velocity_consistency(position_seq: np.ndarray) -> float:
    """
    Velocity consistency score. Low std of velocity magnitudes = smoother.
    Returns score in [0, 1].
    """
    if len(position_seq) < 2:
        return 0.5

    vel = compute_velocity(position_seq)
    vel_magnitude = np.linalg.norm(vel, axis=-1)  # (frames-1, 17)
    consistency = float(np.mean(np.std(vel_magnitude, axis=0)))

    score = 1.0 / (1.0 + consistency * 5.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_acceleration_smoothness(position_seq: np.ndarray) -> float:
    """Low acceleration variance -> smoother movement. Returns score in [0,1]."""
    if len(position_seq) < 3:
        return 0.5

    acc = compute_acceleration(position_seq)
    acc_var = float(np.mean(np.var(acc, axis=0)))
    score = 1.0 / (1.0 + acc_var * 8.0)
    return float(np.clip(score, 0.0, 1.0))


# ------------------------------------------------------------------
# Feature extraction for LSTM
# ------------------------------------------------------------------

def extract_fluidity_features(position_seq: np.ndarray, target_frames: int = 48) -> np.ndarray:
    """
    Extract velocity + acceleration per joint as LSTM input features.

    Parameters
    ----------
    position_seq : (frames, 17, 2)
    target_frames : int  — desired sequence length for LSTM

    Returns
    -------
    features : (target_frames, 17*4)
               [vel_x, vel_y, acc_x, acc_y] per joint
    """
    if len(position_seq) < 3:
        return np.zeros((target_frames, 17 * 4))

    vel = compute_velocity(position_seq)           # (F-1, 17, 2)
    acc = compute_acceleration(position_seq)        # (F-2, 17, 2)

    # Align lengths: trim to shortest
    min_len = min(len(vel), len(acc))
    vel = vel[:min_len]
    acc = acc[:min_len]

    features = np.concatenate([vel, acc], axis=-1)  # (min_len, 17, 4)
    features = features.reshape(min_len, -1)         # (min_len, 68)

    # Resample to target_frames
    if min_len != target_frames:
        indices = np.linspace(0, min_len - 1, target_frames)
        resampled = np.zeros((target_frames, features.shape[1]))
        for i, idx in enumerate(indices):
            lo = int(np.floor(idx))
            hi = min(lo + 1, min_len - 1)
            w = idx - lo
            resampled[i] = (1 - w) * features[lo] + w * features[hi]
        features = resampled

    return features


# ------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------

class LSTMFluididtyModel:
    """
    LSTM-based fluidity regressor.

    Input  : (batch, target_frames, 17*4)  velocity + acceleration features
    Output : (batch, 1)  fluidity score in [0, 1]
    """

    def __init__(self, target_frames: int = 48, feature_dim: int = 68):
        self.target_frames = target_frames
        self.feature_dim = feature_dim
        self.model: Optional[object] = None
        self.is_trained = False

        if TF_AVAILABLE:
            self._build()

    def _build(self):
        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.target_frames, self.feature_dim)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae'],
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train on (N, target_frames, feature_dim) inputs with scalar labels y.

        y = proxy fluidity label derived from rule-based jerk score.
        """
        if self.model is None:
            raise RuntimeError("TensorFlow not available")

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, monitor='val_loss'
                )
            ],
        )
        self.is_trained = True
        return {
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history.get('val_loss', [0])[-1]),
            'epochs_trained': len(history.history['loss']),
        }

    def predict(self, X: np.ndarray) -> float:
        """Predict fluidity score for a single sequence."""
        if self.model is None or not self.is_trained:
            return 0.5
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        return float(self.model.predict(X, verbose=0)[0, 0])

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        import os
        if TF_AVAILABLE and os.path.exists(path):
            self.model = keras.models.load_model(path)
            self.is_trained = True


# ------------------------------------------------------------------
# Main Fluidity Analyzer
# ------------------------------------------------------------------

class FluididtyAnalyzer:
    """
    Combines rule-based and LSTM fluidity analysis.

    Usage:
        analyzer = FluididtyAnalyzer()
        result = analyzer.analyze(normalized_seq)
        # Always works (rule-based).

        # Optional: train LSTM on dataset
        analyzer.train_lstm(sequences, labels)
        result = analyzer.analyze(normalized_seq)  # now includes lstm_score
    """

    def __init__(self, target_frames: int = 48):
        self.target_frames = target_frames
        self.lstm_model = LSTMFluididtyModel(target_frames=target_frames)

    # ------------------------------------------------------------------
    def analyze(self, normalized_seq: np.ndarray) -> Dict:
        """
        Compute fluidity metrics for a normalised keypoint sequence.

        Parameters
        ----------
        normalized_seq : np.ndarray  (frames, 17, 2)

        Returns
        -------
        dict with keys:
            jerk_score            float  [0,1]  — higher is smoother
            velocity_consistency  float  [0,1]
            acceleration_score    float  [0,1]
            rule_based_fluidity   float  [0,1]  — weighted combination
            lstm_score            float  [0,1]  — only if LSTM trained
            overall_fluidity      float  [0,1]  — final score
            interpretation        str
        """
        jerk  = compute_jerk_score(normalized_seq)
        vel_c = compute_velocity_consistency(normalized_seq)
        acc_s = compute_acceleration_smoothness(normalized_seq)

        # Weighted rule-based score
        rule_score = 0.4 * jerk + 0.35 * vel_c + 0.25 * acc_s

        result: Dict = {
            'jerk_score': round(jerk, 4),
            'velocity_consistency': round(vel_c, 4),
            'acceleration_score': round(acc_s, 4),
            'rule_based_fluidity': round(rule_score, 4),
        }

        # LSTM score (if model is trained)
        if self.lstm_model.is_trained:
            features = extract_fluidity_features(normalized_seq, self.target_frames)
            lstm_score = self.lstm_model.predict(features)
            result['lstm_score'] = round(lstm_score, 4)
            overall = 0.5 * rule_score + 0.5 * lstm_score
        else:
            result['lstm_score'] = None
            overall = rule_score

        result['overall_fluidity'] = round(float(overall), 4)
        result['interpretation'] = self._interpret(result['overall_fluidity'])
        return result

    def train_lstm(
        self,
        sequences: list,
        epochs: int = 50,
    ) -> Dict:
        """
        Generate proxy fluidity labels from rule-based scores and train LSTM.

        Parameters
        ----------
        sequences : list of np.ndarray  (frames, 17, 2)
        epochs    : int
        """
        if not TF_AVAILABLE:
            print("  [!] TensorFlow not available — skipping LSTM training.")
            return {}

        print("  Generating proxy fluidity labels from rule-based scores...")
        X, y = [], []
        for seq in sequences:
            if len(seq) < 4:
                continue
            label = self.analyze(seq)['rule_based_fluidity']
            features = extract_fluidity_features(seq, self.target_frames)
            X.append(features)
            y.append(label)

        if len(X) < 10:
            print("  [!] Not enough samples for LSTM training (need ≥10).")
            return {}

        X = np.array(X)
        y = np.array(y)
        print(f"  Training LSTM fluidity model on {len(X)} samples...")
        history = self.lstm_model.train(X, y, epochs=epochs)
        print(f"  [OK] LSTM trained: final loss={history['final_loss']:.4f}")
        return history

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 0.80:
            return "Excellent — smooth, coordinated movement"
        elif score >= 0.60:
            return "Good — minor hesitations detected"
        elif score >= 0.40:
            return "Fair — noticeable jerkiness, improvement needed"
        else:
            return "Poor — significant movement irregularities"


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing LSTM Fluidity Analysis Module...")
    np.random.seed(1)

    # Simulate a smooth sequence (sine-wave motion)
    frames = 60
    t = np.linspace(0, 2 * np.pi, frames)
    smooth_seq = np.zeros((frames, 17, 2))
    for j in range(17):
        smooth_seq[:, j, 0] = 0.3 * np.sin(t + j * 0.2)
        smooth_seq[:, j, 1] = 0.2 * np.cos(t + j * 0.1)

    # Noisy/jerky sequence
    jerky_seq = smooth_seq + np.random.normal(0, 0.05, smooth_seq.shape)

    analyzer = FluididtyAnalyzer()

    res_smooth = analyzer.analyze(smooth_seq)
    res_jerky  = analyzer.analyze(jerky_seq)

    print(f"\n  Smooth sequence:")
    print(f"    jerk_score:          {res_smooth['jerk_score']:.4f}")
    print(f"    velocity_consistency:{res_smooth['velocity_consistency']:.4f}")
    print(f"    overall_fluidity:    {res_smooth['overall_fluidity']:.4f}")
    print(f"    interpretation:      {res_smooth['interpretation']}")

    print(f"\n  Jerky sequence:")
    print(f"    jerk_score:          {res_jerky['jerk_score']:.4f}")
    print(f"    velocity_consistency:{res_jerky['velocity_consistency']:.4f}")
    print(f"    overall_fluidity:    {res_jerky['overall_fluidity']:.4f}")
    print(f"    interpretation:      {res_jerky['interpretation']}")

    print("\n  [OK] Fluidity module OK")
