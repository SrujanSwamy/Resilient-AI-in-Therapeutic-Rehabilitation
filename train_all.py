"""
Train CNN Model on All 6 Exercises
----------------------------------
Trains a single CNN classifier on all exercises (Ex1-Ex6) combined.
Saves the trained model for later use.

Usage:
    python train_all.py                    # Train with default settings
    python train_all.py --epochs 200       # Train 200 epochs (as per paper)
    python train_all.py --evaluate         # Train and evaluate
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add subteam1_edge to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'subteam1_edge'))

from normalization import SkeletonNormalizer
from comparison import MovementComparator
from confidence import ConfidenceScorer, CONFIDENCE_THRESHOLD

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("ERROR: TensorFlow not available. Install with: pip install tensorflow")
    sys.exit(1)


# Model save path
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_classifier_all_exercises.keras")
REFERENCE_PATH = os.path.join(MODEL_DIR, "reference_all_exercises.npz")


class MultiExerciseTrainer:
    """
    Trains CNN classifier on all 6 exercises combined.
    
    As per research paper:
    - CNN layer on MoveNet output
    - Binary classification (correct/incorrect)
    - 200 epochs training
    - Target: 99.29% train, 99.88% test accuracy
    """
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.normalizer = SkeletonNormalizer()
        self.target_frames = 50
        self.num_keypoints = 17
        
        # Model
        self.model = None
        
        # Data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Reference models per exercise
        self.reference_models = {}
    
    def load_segmentation(self):
        """Load segmentation CSV."""
        seg_path = os.path.join(self.dataset_path, "Segmentation.csv")
        return pd.read_csv(seg_path, sep=';')
    
    def load_sample(self, exercise_id: str, video_id: str, 
                    first_frame: int, last_frame: int,
                    camera: str = "c17", fps: int = 30):
        """Load and preprocess a single sample."""
        filename = f"{video_id}-{camera}-{fps}fps.npy"
        filepath = os.path.join(self.dataset_path, "2d_joints", exercise_id, filename)
        
        if not os.path.exists(filepath):
            return None
        
        data = np.load(filepath)
        
        # Extract segment
        first_frame = max(0, first_frame)
        last_frame = min(len(data), last_frame)
        segment = data[first_frame:last_frame]
        
        if len(segment) < 5:
            return None
        
        # Convert to 17 keypoints
        if segment.shape[1] >= 17:
            keypoints = segment[:, :17, :2]
        else:
            keypoints = np.zeros((len(segment), 17, 2))
            keypoints[:, :segment.shape[1], :] = segment[:, :, :2]
        
        # Add confidence
        confidence = np.ones((len(keypoints), 17, 1))
        keypoints_with_conf = np.concatenate([keypoints, confidence], axis=2)
        
        # Normalize
        normalized = self.normalizer.normalize_sequence(keypoints_with_conf)
        
        # Resample
        resampled = self._resample(normalized[:, :, :2], self.target_frames)
        
        return resampled
    
    def _resample(self, sequence, target_length):
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
    
    def load_all_data(self, camera: str = "c17", fps: int = 30, train_split: float = 0.8):
        """Load data from all 6 exercises."""
        print("\n" + "="*60)
        print("Loading data from ALL 6 exercises")
        print("="*60)
        
        seg_df = self.load_segmentation()
        
        X_all = []
        y_all = []
        exercise_counts = {i: {'correct': 0, 'incorrect': 0} for i in range(1, 7)}
        
        for exercise_num in range(1, 7):
            exercise_id = f"Ex{exercise_num}"
            print(f"\nLoading {exercise_id}...")
            
            exercise_df = seg_df[seg_df['exercise_id'] == exercise_num]
            
            for _, row in exercise_df.iterrows():
                sample = self.load_sample(
                    exercise_id,
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
                
                if label == 1:
                    exercise_counts[exercise_num]['correct'] += 1
                else:
                    exercise_counts[exercise_num]['incorrect'] += 1
            
            correct = exercise_counts[exercise_num]['correct']
            incorrect = exercise_counts[exercise_num]['incorrect']
            print(f"  {exercise_id}: {correct} correct, {incorrect} incorrect")
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        print(f"\n{'='*60}")
        print(f"Total samples loaded: {len(X_all)}")
        print(f"  Correct: {np.sum(y_all == 1)}")
        print(f"  Incorrect: {np.sum(y_all == 0)}")
        print(f"{'='*60}")
        
        # Shuffle and split
        indices = np.random.permutation(len(X_all))
        n_train = int(len(X_all) * train_split)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        self.X_train = X_all[train_idx]
        self.y_train = y_all[train_idx]
        self.X_val = X_all[val_idx]
        self.y_val = y_all[val_idx]
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Validation set: {len(self.X_val)} samples")
        
        return exercise_counts
    
    def build_model(self):
        """Build CNN model architecture as per research paper."""
        print("\nBuilding CNN model...")
        
        input_shape = (self.target_frames, self.num_keypoints, 2)
        
        self.model = keras.Sequential([
            # Input layer
            keras.layers.InputLayer(input_shape=input_shape),
            
            # Reshape for Conv1D
            keras.layers.Reshape((self.target_frames, self.num_keypoints * 2)),
            
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
            
            # Dense layers
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
    
    def train(self, epochs: int = 200, batch_size: int = 32):
        """Train the CNN model."""
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        if self.model is None:
            self.build_model()
        
        print(f"\n{'='*60}")
        print(f"Training CNN Classifier ({epochs} epochs)")
        print(f"{'='*60}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final metrics
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Final Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Final Val Accuracy: {val_acc*100:.2f}%")
        print(f"  Epochs trained: {len(history.history['accuracy'])}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'epochs': len(history.history['accuracy']),
            'history': history.history
        }
    
    def save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        # Create models directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model
        self.model.save(MODEL_PATH)
        print(f"\n[✓] Model saved to: {MODEL_PATH}")
        
        return MODEL_PATH
    
    def load_model(self):
        """Load pre-trained model from disk."""
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            print(f"[✓] Model loaded from: {MODEL_PATH}")
            return True
        else:
            print(f"[!] No saved model found at: {MODEL_PATH}")
            return False
    
    def evaluate(self):
        """Evaluate model on validation set."""
        if self.model is None:
            raise ValueError("No model loaded.")
        
        if self.X_val is None:
            raise ValueError("No validation data. Call load_all_data() first.")
        
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        
        # Evaluate
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        
        print(f"  Validation Loss: {loss:.4f}")
        print(f"  Validation Accuracy: {accuracy*100:.2f}%")
        
        # Per-class metrics
        predictions = (self.model.predict(self.X_val, verbose=0) >= 0.5).astype(int).flatten()
        
        correct_as_correct = np.sum((predictions == 1) & (self.y_val == 1))
        correct_as_incorrect = np.sum((predictions == 0) & (self.y_val == 1))
        incorrect_as_correct = np.sum((predictions == 1) & (self.y_val == 0))
        incorrect_as_incorrect = np.sum((predictions == 0) & (self.y_val == 0))
        
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Correct  Incorrect")
        print(f"  Actual Correct    {correct_as_correct:4d}      {correct_as_incorrect:4d}")
        print(f"  Actual Incorrect  {incorrect_as_correct:4d}      {incorrect_as_incorrect:4d}")
        
        # Precision, Recall, F1
        precision = correct_as_correct / (correct_as_correct + incorrect_as_correct + 1e-8)
        recall = correct_as_correct / (correct_as_correct + correct_as_incorrect + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"\nMetrics (for 'Correct' class):")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall: {recall*100:.2f}%")
        print(f"  F1 Score: {f1*100:.2f}%")
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, sample):
        """Predict single sample."""
        if self.model is None:
            raise ValueError("No model loaded.")
        
        if sample.ndim == 3:
            sample = np.expand_dims(sample, axis=0)
        
        proba = self.model.predict(sample, verbose=0).flatten()
        prediction = (proba >= 0.5).astype(int)
        
        return prediction[0], proba[0]


def main():
    parser = argparse.ArgumentParser(description="Train CNN on all 6 exercises")
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: 200)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset path')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--load', action='store_true', help='Load existing model instead of training')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  CNN CLASSIFIER TRAINING - ALL 6 EXERCISES")
    print("  Based on: 'Resilient AI in Therapeutic Rehabilitation'")
    print("="*70)
    
    trainer = MultiExerciseTrainer(args.dataset)
    
    if args.load:
        # Load existing model
        if trainer.load_model():
            trainer.load_all_data()
            trainer.evaluate()
        else:
            print("Train a model first with: python train_all.py")
    else:
        # Train new model
        trainer.load_all_data()
        trainer.train(epochs=args.epochs)
        trainer.save_model()
        
        if args.evaluate:
            trainer.evaluate()
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
