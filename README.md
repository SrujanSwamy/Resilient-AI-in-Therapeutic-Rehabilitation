# Rehabilitation Exercise Analysis System

Implementation of **"Resilient AI in Therapeutic Rehabilitation"** research paper.

## Project Structure

```
da project/
├── dataset/                    # REHAB24-6 dataset
│   ├── 2d_joints/             # Joint coordinates (Ex1-Ex6)
│   ├── videos/                # Exercise video recordings
│   └── Segmentation.csv       # Metadata with correctness labels
│
├── subteam1_edge/             # Module 1-5 Implementation
│   ├── movenet.py             # Module 1: MoveNet pose estimation (17 keypoints)
│   ├── normalization.py       # Module 2: Skeleton normalization (hip center + torso scale)
│   ├── reference_model.py     # Module 3: Reference model from correct samples
│   ├── comparison.py          # Module 4: Euclidean, MSE, RMSE metrics
│   ├── confidence.py          # Module 5a: Confidence scoring (0.8 threshold)
│   └── classifier.py          # Module 5b: CNN + Threshold classification
│
├── main.py                    # Complete pipeline integration
├── show_dataset.py            # Dataset visualization utility
├── view_database.py           # Database viewer utility
├── requirements.txt           # Python dependencies
└── PPT_CONTENT_MIDSEM.md      # Presentation content
```

## Modules (As Per Research Paper)

### Module 1: MoveNet Pose Estimation
- **Model**: MoveNet Thunder (SinglePose)
- **Input**: 256×256 RGB image
- **Output**: 17 keypoints with (x, y, confidence)
- **Keypoints**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

### Module 2: Skeleton Normalization
- **Purpose**: Scale and position invariance for subject comparison
- **Method**: Center at hip, scale by torso length
- **Formula**: P_normalized = (P_original - hip_center) / torso_length

### Module 3: Reference Model Building
- **Source**: Correct execution samples (correctness = 1)
- **Output**: Mean template + standard deviation bounds
- **Alignment**: Temporal resampling to fixed frame count

### Module 4: Movement Comparison
- **Euclidean Distance**: C = ||Y_correct - Y_actual||
- **MSE**: (1/N) × Σ(y_correct - y_actual)²
- **RMSE**: √MSE
- **Per-joint analysis**: Identify most deviant joints

### Module 5: Confidence & Classification
- **Confidence Threshold**: 0.8 (as per paper)
- **CNN Classifier**: Conv1D layers on pose sequence
- **Threshold Classifier**: RMSE-based binary decision
- **Paper Results**: 99.29% train, 99.88% test accuracy

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python main.py --exercise Ex1
```

### Run with Full Evaluation
```bash
python main.py --exercise Ex1 --evaluate
```

### Run Specific Module
```bash
python main.py --exercise Ex1 --module 5   # Module 5 only
```

### View Dataset
```bash
python show_dataset.py
```

## Sample Output

```
======================================================================
  REHABILITATION EXERCISE ANALYSIS SYSTEM
  Based on: 'Resilient AI in Therapeutic Rehabilitation'
======================================================================
  TensorFlow Available: True
  Confidence Threshold: 0.8
  MoveNet Keypoints: 17
======================================================================

MODULE 1: MoveNet Pose Estimation
  [✓] MoveNet Thunder loaded successfully
  [✓] Input size: 256x256
  [✓] Output: 17 keypoints (x, y, confidence)

MODULE 2: Skeleton Normalization
  [✓] Normalization complete

MODULE 3: Reference Model Building
  [✓] Reference model built for Ex1
  [✓] Samples used: 90
  [✓] Reference shape: (50, 17, 2)

MODULE 4: Movement Comparison
  Correct Sample: RMSE: 0.2482
  Incorrect Sample: RMSE: 0.3105
  [✓] Comparison metrics computed

MODULE 5: Confidence Scoring & Classification
  [✓] Training Complete
      CNN Train Acc: 99.3%
      CNN Val Acc: 91.7%
      Threshold: 0.3829

======================================================================
  PIPELINE COMPLETE
======================================================================
```

## Research Paper Reference

- **Paper**: "Resilient AI in Therapeutic Rehabilitation"
- **Method**: MoveNet pose estimation + CNN classification
- **Confidence Threshold**: 0.8 for reliable predictions
- **Visual Feedback**: Green (correct) / Blue (incorrect) skeleton
