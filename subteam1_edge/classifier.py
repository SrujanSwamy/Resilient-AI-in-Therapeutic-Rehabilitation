"""
Module 5: Classification (CNN + Threshold)
------------------------------------------
Exercise correctness classification as per research paper.
Implements BOTH CNN-based and threshold-based classification.

Research Paper Reference:
    "A CNN layer is added on top of MoveNet output for classification.
    The system classifies movements as correct or incorrect based on
    deviation from reference template and confidence scoring."

Classification Methods:
    1. CNN Classifier: Neural network on pose features
    2. Threshold Classifier: RMSE threshold comparison

Paper Results:
    - Train Accuracy: 99.29%
    - Test Accuracy: 99.88%
    - 200 epochs training
"""

import numpy as np
import os
import pandas as pd
from typing import Dict, Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from normalization import SkeletonNormalizer
from reference_model import ReferenceModel
from comparison import MovementComparator, compute_rmse
from confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD
from imputation import JointImputer
from compensation import CompensationDetector
from fluidity import FluididtyAnalyzer


class CNNClassifier:
    """
    CNN-based classifier on MoveNet pose features.
    
    As per research paper:
    - CNN layer added on MoveNet output
    - Binary classification (correct/incorrect)
    - 99.88% test accuracy target
    
    Architecture:
        Input: (frames, 17, 2) normalized keypoints
        -> Flatten
        -> Dense layers
        -> Binary output
    """
    
    def __init__(self, num_frames: int = 50, num_keypoints: int = 17):
        """
        Initialize CNN classifier.
        
        Parameters:
            num_frames: Number of frames per sample
            num_keypoints: Number of keypoints (17 for MoveNet)
        """
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.input_shape = (num_frames, num_keypoints, 2)
        self.model = None
        
        if TF_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """
        Build CNN model architecture.
        
        Simple architecture for pose classification:
        - Conv layers for spatial-temporal features
        - Dense layers for classification
        """
        self.model = keras.Sequential([
            # Input layer
            keras.layers.InputLayer(input_shape=self.input_shape),
            
            # Reshape for Conv1D (treat frames as sequence)
            keras.layers.Reshape((self.num_frames, self.num_keypoints * 2)),
            
            # Conv1D layers for temporal patterns
            keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            
            keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            
            keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers for classification
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer - binary classification
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 200, batch_size: int = 32) -> Dict:
        """
        Train CNN classifier.
        
        Parameters:
            X_train: Training data (samples, frames, 17, 2)
            y_train: Training labels (samples,) - 0 or 1
            X_val: Validation data [optional]
            y_val: Validation labels [optional]
            epochs: Number of training epochs (200 as per paper)
            batch_size: Batch size
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            raise RuntimeError("TensorFlow not available")
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=20,
                    restore_best_weights=True
                )
            ]
        )
        
        return {
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history.get('val_accuracy', [None])[-1],
            'epochs_trained': len(history.history['accuracy'])
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict correctness.
        
        Parameters:
            X: Input data (samples, frames, 17, 2) or (frames, 17, 2)
            
        Returns:
            tuple: (binary_predictions, probabilities)
        """
        if self.model is None:
            raise RuntimeError("TensorFlow not available")
        
        # Handle single sample
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        
        proba = self.model.predict(X, verbose=0).flatten()
        predictions = (proba >= 0.5).astype(int)
        
        return predictions, proba
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"  [✓] CNN model saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        if TF_AVAILABLE and os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"  [✓] CNN model loaded from: {filepath}")
            return True
        return False


class ThresholdClassifier:
    """
    Threshold-based classifier using RMSE comparison.
    
    Classification rule:
    - If RMSE < threshold: Correct (1)
    - If RMSE >= threshold: Incorrect (0)
    
    The threshold is learned from training data statistics.
    """
    
    def __init__(self):
        """Initialize threshold classifier."""
        self.rmse_threshold = None
        self.reference_mean = None
        self.reference_std = None
    
    def train(self, correct_rmse_values: np.ndarray, incorrect_rmse_values: np.ndarray) -> Dict:
        """
        Learn optimal threshold from training data.
        
        Sets threshold as midpoint between correct and incorrect RMSE distributions.
        
        Parameters:
            correct_rmse_values: RMSE values for correct samples
            incorrect_rmse_values: RMSE values for incorrect samples
            
        Returns:
            dict: Training statistics
        """
        correct_mean = np.mean(correct_rmse_values)
        incorrect_mean = np.mean(incorrect_rmse_values)
        
        # Threshold at midpoint
        self.rmse_threshold = (correct_mean + incorrect_mean) / 2
        
        return {
            'correct_rmse_mean': correct_mean,
            'incorrect_rmse_mean': incorrect_mean,
            'threshold': self.rmse_threshold
        }
    
    def predict(self, rmse: float) -> Tuple[int, float]:
        """
        Classify based on RMSE threshold.
        
        Parameters:
            rmse: RMSE value from comparison
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.rmse_threshold is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Lower RMSE = more likely correct
        prediction = 1 if rmse < self.rmse_threshold else 0
        
        # Confidence based on distance from threshold
        distance = abs(rmse - self.rmse_threshold)
        confidence = 1.0 / (1.0 + distance)
        
        return prediction, confidence


class ExerciseClassifier:
    """
    Complete exercise classification pipeline.
    
    Combines all modules:
    - Module 1: MoveNet pose estimation
    - Module 2: Skeleton normalization
    - Module 3: Reference model building
    - Module 4: Movement comparison
    - Module 5: Confidence + Classification (CNN + Threshold)
    
    As per research paper methodology.
    """
    
    def __init__(self, exercise_id: str, dataset_path: str = "dataset", 
                 use_cnn: bool = True, use_threshold: bool = True):
        """
        Initialize classifier.
        
        Parameters:
            exercise_id: Exercise folder (Ex1, Ex2, etc.)
            dataset_path: Path to REHAB24-6 dataset
            use_cnn: Enable CNN classifier
            use_threshold: Enable threshold classifier
        """
        self.exercise_id = exercise_id
        self.dataset_path = dataset_path
        self.use_cnn = use_cnn and TF_AVAILABLE
        self.use_threshold = use_threshold
        
        # Module instances
        self.normalizer = SkeletonNormalizer()
        self.reference_model = ReferenceModel(dataset_path)
        self.comparator = None
        self.confidence_scorer = ConfidenceScorer()

        # End-Semester enhancements
        self.imputer = JointImputer(confidence_threshold=0.3)
        self.compensation_detector = CompensationDetector(severity_threshold=0.3)
        self.fluidity_analyzer = FluididtyAnalyzer(target_frames=48)

        # Classifiers
        self.cnn_classifier = None
        self.threshold_classifier = ThresholdClassifier() if use_threshold else None

        # Training state
        self.is_trained = False
        self.target_frames = 50
    
    def load_sample(self, video_id: str, first_frame: int, last_frame: int,
                    camera: str = "c17", fps: int = 30) -> Optional[np.ndarray]:
        """
        Load and preprocess a single sample.
        
        Parameters:
            video_id: Video identifier
            first_frame: Start frame
            last_frame: End frame
            camera: Camera view
            fps: Frame rate
            
        Returns:
            ndarray: Normalized keypoints (target_frames, 17, 2)
        """
        # Get exercise_id format
        if isinstance(self.exercise_id, str) and self.exercise_id.startswith("Ex"):
            exercise_folder = self.exercise_id
        else:
            exercise_folder = f"Ex{self.exercise_id}"
        
        # Load keypoints
        filename = f"{video_id}-{camera}-{fps}fps.npy"
        filepath = os.path.join(self.dataset_path, "2d_joints", exercise_folder, filename)
        
        if not os.path.exists(filepath):
            return None

        data = np.load(filepath)

        # Extract segment
        first_frame = max(0, first_frame)
        last_frame = min(len(data), last_frame)
        segment = data[first_frame:last_frame]

        if len(segment) < 5:
            return None

        # Convert to 17 keypoints with confidence
        if segment.shape[1] >= 17:
            keypoints = segment[:, :17, :2]
        else:
            keypoints = np.zeros((len(segment), 17, 2))
            keypoints[:, :segment.shape[1], :] = segment[:, :, :2]

        # Add confidence
        confidence = np.ones((len(keypoints), 17, 1))
        keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)

        # --- End-Sem Enhancement: Impute missing joints ---
        keypoints_with_conf, _ = self.imputer.impute_sequence(keypoints_with_conf)

        # Normalize
        normalized = self.normalizer.normalize_sequence(keypoints_with_conf)

        # Resample to target length
        resampled = self._resample(normalized[:, :, :2], self.target_frames)

        return resampled
    
    def _resample(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Resample sequence to fixed length."""
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
    
    def train(self, camera: str = "c17", fps: int = 30, 
              train_split: float = 0.8, epochs: int = 200) -> Dict:
        """
        Train both CNN and threshold classifiers.
        
        Parameters:
            camera: Camera view to use
            fps: Frame rate version
            train_split: Fraction for training (rest for validation)
            epochs: CNN training epochs (200 as per paper)
            
        Returns:
            dict: Training results
        """
        print(f"\n{'='*60}")
        print(f"Training Classifier for {self.exercise_id}")
        print(f"{'='*60}")
        
        # Build reference model first
        ref_stats = self.reference_model.build_from_samples(
            self.exercise_id, camera, fps, self.target_frames
        )
        
        # Initialize comparator with reference
        self.comparator = MovementComparator(
            self.reference_model.reference_mean,
            self.reference_model.reference_std
        )
        
        # Load all samples
        seg_df = self.reference_model.load_segmentation()
        
        # Get exercise number
        if isinstance(self.exercise_id, str) and self.exercise_id.startswith("Ex"):
            exercise_num = int(self.exercise_id[2:])
        else:
            exercise_num = int(self.exercise_id)
        
        exercise_df = seg_df[seg_df['exercise_id'] == exercise_num].copy()
        
        X_all = []
        y_all = []
        rmse_correct = []
        rmse_incorrect = []
        
        print(f"Loading {len(exercise_df)} samples...")
        
        for _, row in exercise_df.iterrows():
            sample = self.load_sample(
                row['video_id'], 
                int(row['first_frame']), 
                int(row['last_frame']),
                camera, fps
            )
            
            if sample is None:
                continue
            
            label = int(row['correctness'])
            
            X_all.append(sample)
            y_all.append(label)
            
            # Compute RMSE for threshold classifier
            metrics = self.comparator.compare_sequence(sample)
            
            if label == 1:
                rmse_correct.append(metrics['overall_rmse'])
            else:
                rmse_incorrect.append(metrics['overall_rmse'])
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        print(f"Loaded {len(X_all)} samples")
        print(f"  Correct: {np.sum(y_all == 1)}, Incorrect: {np.sum(y_all == 0)}")
        
        results = {
            'total_samples': len(X_all),
            'correct_samples': int(np.sum(y_all == 1)),
            'incorrect_samples': int(np.sum(y_all == 0))
        }
        
        # Split data
        n_train = int(len(X_all) * train_split)
        indices = np.random.permutation(len(X_all))
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        # Train threshold classifier
        if self.use_threshold and len(rmse_correct) > 0 and len(rmse_incorrect) > 0:
            print("\nTraining Threshold Classifier...")
            threshold_stats = self.threshold_classifier.train(
                np.array(rmse_correct), np.array(rmse_incorrect)
            )
            results['threshold_classifier'] = threshold_stats
            print(f"  Threshold: {threshold_stats['threshold']:.4f}")
        
        # Train CNN classifier
        if self.use_cnn:
            print(f"\nTraining CNN Classifier ({epochs} epochs)...")
            self.cnn_classifier = CNNClassifier(self.target_frames, 17)
            cnn_stats = self.cnn_classifier.train(
                X_train, y_train, X_val, y_val, epochs=epochs
            )
            results['cnn_classifier'] = cnn_stats
            print(f"  Train Accuracy: {cnn_stats['train_accuracy']*100:.2f}%")
            if cnn_stats['val_accuracy']:
                print(f"  Val Accuracy: {cnn_stats['val_accuracy']*100:.2f}%")
        
        self.is_trained = True
        
        return results
    
    def predict(self, sample: np.ndarray) -> Dict:
        """
        Classify a single sample using both methods.
        
        Parameters:
            sample: Normalized keypoints (frames, 17, 2)
            
        Returns:
            dict: Classification results from both classifiers
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Ensure correct shape
        if sample.shape[0] != self.target_frames:
            sample = self._resample(sample, self.target_frames)
        
        results = {}
        
        # Compute RMSE
        metrics = self.comparator.compare_sequence(sample)
        results['rmse'] = metrics['overall_rmse']
        results['mean_rmse'] = metrics['mean_rmse']
        
        # Confidence scoring
        confidence = self.confidence_scorer.score_from_rmse(metrics['overall_rmse'])
        results['confidence'] = confidence
        results['is_reliable'] = confidence >= CONFIDENCE_THRESHOLD
        
        # Threshold classification
        if self.use_threshold and self.threshold_classifier.rmse_threshold is not None:
            pred, conf = self.threshold_classifier.predict(metrics['overall_rmse'])
            results['threshold_prediction'] = pred
            results['threshold_confidence'] = conf
        
        # CNN classification
        if self.use_cnn and self.cnn_classifier is not None:
            preds, proba = self.cnn_classifier.predict(sample)
            results['cnn_prediction'] = int(preds[0])
            results['cnn_probability'] = float(proba[0])
        
        # Final prediction (prefer CNN if available, else threshold)
        if 'cnn_prediction' in results:
            results['final_prediction'] = results['cnn_prediction']
            results['prediction_method'] = 'CNN'
        elif 'threshold_prediction' in results:
            results['final_prediction'] = results['threshold_prediction']
            results['prediction_method'] = 'Threshold'
        else:
            results['final_prediction'] = None
            results['prediction_method'] = 'None'

        results['label'] = 'Correct' if results['final_prediction'] == 1 else 'Incorrect'

        # --- End-Sem Enhancement: Compensation Detection ---
        ref_mean = None
        if self.comparator is not None:
            ref_mean = self.comparator.reference_mean
        comp_result = self.compensation_detector.detect(sample, self.exercise_id, ref_mean)
        results['compensation_found'] = comp_result['compensation_found']
        results['compensation_types'] = comp_result['types']
        results['compensation_severity'] = comp_result['severity']
        results['compensation_feedback'] = self.compensation_detector.get_feedback_message(comp_result)

        # --- End-Sem Enhancement: Fluidity Analysis ---
        fluidity_result = self.fluidity_analyzer.analyze(sample)
        results['fluidity_score'] = fluidity_result['overall_fluidity']
        results['fluidity_jerk'] = fluidity_result['jerk_score']
        results['fluidity_velocity'] = fluidity_result['velocity_consistency']
        results['fluidity_interpretation'] = fluidity_result['interpretation']

        return results
    
    def evaluate(self, camera: str = "c17", fps: int = 30) -> Dict:
        """
        Evaluate classifier on all samples.
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        print(f"\n{'='*60}")
        print(f"Evaluating Classifier for {self.exercise_id}")
        print(f"{'='*60}")
        
        # Load all samples
        seg_df = self.reference_model.load_segmentation()
        
        if isinstance(self.exercise_id, str) and self.exercise_id.startswith("Ex"):
            exercise_num = int(self.exercise_id[2:])
        else:
            exercise_num = int(self.exercise_id)
        
        exercise_df = seg_df[seg_df['exercise_id'] == exercise_num].copy()
        
        cnn_correct = 0
        threshold_correct = 0
        total = 0
        
        for _, row in exercise_df.iterrows():
            sample = self.load_sample(
                row['video_id'],
                int(row['first_frame']),
                int(row['last_frame']),
                camera, fps
            )
            
            if sample is None:
                continue
            
            true_label = int(row['correctness'])
            
            pred_results = self.predict(sample)
            
            if 'cnn_prediction' in pred_results:
                if pred_results['cnn_prediction'] == true_label:
                    cnn_correct += 1
            
            if 'threshold_prediction' in pred_results:
                if pred_results['threshold_prediction'] == true_label:
                    threshold_correct += 1
            
            total += 1
        
        results = {
            'total_samples': total
        }
        
        if self.use_cnn:
            results['cnn_accuracy'] = cnn_correct / total if total > 0 else 0
            print(f"CNN Accuracy: {results['cnn_accuracy']*100:.2f}%")
        
        if self.use_threshold:
            results['threshold_accuracy'] = threshold_correct / total if total > 0 else 0
            print(f"Threshold Accuracy: {results['threshold_accuracy']*100:.2f}%")
        
        return results


if __name__ == "__main__":
    # Test classification module
    print("Testing Classification Module...")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        # Test CNN model building
        cnn = CNNClassifier(num_frames=50, num_keypoints=17)
        print(f"CNN model built: {cnn.model is not None}")
        if cnn.model is not None:
            cnn.model.summary()
    
    # Test threshold classifier
    threshold = ThresholdClassifier()
    correct_rmse = np.array([0.05, 0.06, 0.04, 0.07, 0.05])
    incorrect_rmse = np.array([0.15, 0.18, 0.12, 0.20, 0.16])
    
    stats = threshold.train(correct_rmse, incorrect_rmse)
    print(f"\nThreshold Classifier:")
    print(f"  Correct RMSE mean: {stats['correct_rmse_mean']:.4f}")
    print(f"  Incorrect RMSE mean: {stats['incorrect_rmse_mean']:.4f}")
    print(f"  Learned threshold: {stats['threshold']:.4f}")
    
    # Test predictions
    pred, conf = threshold.predict(0.06)
    print(f"\nTest RMSE=0.06: Prediction={pred}, Confidence={conf:.2f}")
    
    pred, conf = threshold.predict(0.16)
    print(f"Test RMSE=0.16: Prediction={pred}, Confidence={conf:.2f}")
