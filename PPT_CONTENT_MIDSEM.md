# Mid-Semester Presentation Content
## Resilient AI in Therapeutic Rehabilitation

---

# SLIDE 1: Introduction (1/2)

## Project Title
**AI-Powered Rehabilitation Exercise Analysis System**

## Project Overview
- Implementation of computer vision-based rehabilitation monitoring system
- Real-time pose estimation for therapeutic exercise evaluation
- Automated correctness assessment with confidence scoring
- Edge-Cloud distributed architecture for scalable deployment

## Domain
- Healthcare Technology / Physiotherapy
- Computer Vision & Deep Learning
- Human Pose Estimation
- Time-Series Pattern Recognition

## Key Technologies
- TensorFlow / TensorFlow Hub
- MoveNet Thunder (Single-Pose Detection)
- Python (NumPy, OpenCV, Pandas)
- REST API (FastAPI)

---

# SLIDE 2: Introduction (2/2)

## Why This Matters

### The Healthcare Challenge
- 2.4 billion people globally require rehabilitation services
- Only 50% of affected populations have access to rehabilitation
- Physiotherapist shortage in rural and underserved areas
- Manual monitoring is time-intensive and inconsistent

### The Solution
- AI-powered automated exercise monitoring
- Real-time feedback without constant clinician supervision
- Objective, quantifiable assessment of movement quality
- Scalable system deployable on low-cost edge devices

### Target Application
- Home-based rehabilitation programs
- Post-surgical recovery monitoring
- Sports injury rehabilitation
- Elderly care and mobility assistance

---

# SLIDE 3: Problem Statement and Research Objectives

## Problem Statement
Traditional physiotherapy rehabilitation requires continuous supervision by trained clinicians to ensure patients perform exercises correctly. This creates:
1. **Accessibility barriers** - Limited availability of physiotherapists
2. **Scalability issues** - One-to-one supervision model limits patient capacity
3. **Consistency problems** - Subjective human assessment varies between sessions
4. **Cost burden** - Expensive in-person sessions limit treatment duration

## Research Objectives

| # | Objective | Target |
|---|-----------|--------|
| 1 | Real-time pose estimation | 17 keypoints extraction using MoveNet Thunder |
| 2 | Automated correctness classification | Binary classification (correct/incorrect) |
| 3 | Confidence-based reliability | 0.8 threshold for trustworthy predictions |
| 4 | Performance metrics computation | Euclidean distance, MSE, RMSE formulas |
| 5 | Scalable architecture | Edge-cloud distributed system design |

## Success Criteria
- Classification accuracy > 95%
- System latency < 120ms total pipeline
- Confidence threshold correctly filters unreliable predictions

---

# SLIDE 4: Existing Methodology (1/3) - System Architecture

## Research Paper: "Resilient AI in Therapeutic Rehabilitation"

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGE CLIENT                              │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────┐            │
│  │  Camera  │ → │  MoveNet    │ → │ Normalization │            │
│  │  Input   │   │  Thunder    │   │  (17 joints)  │            │
│  └──────────┘   └─────────────┘   └──────────────┘            │
│                       ↓                   ↓                     │
│              ┌──────────────┐    ┌─────────────────┐           │
│              │  Reference   │ ←─ │   Comparison    │           │
│              │    Model     │    │  (Euclidean/MSE)│           │
│              └──────────────┘    └─────────────────┘           │
│                                          ↓                      │
│              ┌──────────────┐    ┌─────────────────┐           │
│              │  Confidence  │ ←─ │  Classification │           │
│              │  (≥ 0.8)     │    │ (Correct/Incorr)│           │
│              └──────────────┘    └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CLOUD SERVER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Database   │  │   REST API   │  │   Dashboard  │         │
│  │   (NoSQL)    │  │   Endpoints  │  │  (Clinician) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components
- **MoveNet Thunder**: TensorFlow-based pose estimation model
- **Edge Processing**: Local analysis for low-latency feedback
- **Cloud Storage**: Centralized data management and analytics

---

# SLIDE 5: Existing Methodology (2/3) - MoveNet & Processing Pipeline

## MoveNet Thunder Pose Estimation

### Model Specifications
| Parameter | Value |
|-----------|-------|
| Model | MoveNet Thunder (SinglePose) |
| Framework | TensorFlow / TensorFlow Hub |
| Input Size | 256 × 256 pixels |
| Output | 17 keypoints with (x, y, confidence) |
| Speed | ~30 FPS on modern hardware |

### 17 Keypoint Structure
```
        nose(0)
    left_eye(1)  right_eye(2)
   left_ear(3)    right_ear(4)
         
left_shoulder(5) ─── right_shoulder(6)
       │                    │
  left_elbow(7)       right_elbow(8)
       │                    │
  left_wrist(9)       right_wrist(10)

   left_hip(11) ─── right_hip(12)
       │                    │
  left_knee(13)       right_knee(14)
       │                    │
 left_ankle(15)      right_ankle(16)
```

### Normalization Process
1. **Find Hip Center**: Midpoint of left_hip and right_hip
2. **Translation**: Shift all joints so hip center becomes origin (0, 0)
3. **Calculate Torso Length**: Distance from hip center to shoulder center
4. **Scaling**: Divide all coordinates by torso length

**Purpose**: Enables comparison between subjects of different body sizes and positions

---

# SLIDE 6: Existing Methodology (3/3) - Comparison & Classification

## Mathematical Formulation

### Euclidean Distance (Correction Vector)
$$C = ||Y_{correct} - Y_{actual}|| = \sqrt{\sum_{i=1}^{17} (y_{correct,i} - y_{actual,i})^2}$$

Where:
- $Y_{correct}$ = Reference keypoint coordinates (from correct samples)
- $Y_{actual}$ = Observed keypoint coordinates (patient movement)
- $C$ = Overall deviation measure

### Mean Squared Error (MSE)
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_{correct,i} - y_{actual,i})^2$$

### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{correct,i} - y_{actual,i})^2}$$

## Confidence Scoring

### Threshold-Based Decision
- **Confidence ≥ 0.8**: Prediction is **reliable** → Proceed to classification
- **Confidence < 0.8**: Prediction is **uncertain** → Display warning to user

### Confidence Computation
$$Confidence = \frac{1}{1 + RMSE}$$

## Classification
| RMSE vs Threshold | Confidence | Result |
|-------------------|------------|--------|
| Low RMSE | ≥ 0.8 | **Correct Execution** (Green) |
| High RMSE | ≥ 0.8 | **Incorrect Execution** (Blue) |
| Any | < 0.8 | **Warning** (Unreliable detection) |

---

# SLIDE 7: Dataset Details

## REHAB24-6 Dataset

### Overview
| Attribute | Value |
|-----------|-------|
| Dataset Name | REHAB24-6 |
| Total Exercises | 6 different rehabilitation exercises |
| Participants | Multiple subjects (varying body types) |
| Camera Views | 2 cameras (c17, c18) |
| Frame Rates | 30 fps, 120 fps |
| Data Types | 2D Joints, 3D Joints, 2D Markers, 3D Markers, Videos |

### Exercise Categories
| Exercise ID | Type | Description |
|-------------|------|-------------|
| Ex1 | Upper Body | Arm raising exercise (right arm) |
| Ex2 | Upper Body | Arm movement variations |
| Ex3 | Upper Body | Shoulder rehabilitation |
| Ex4 | Lower Body | Leg exercises |
| Ex5 | Lower Body | Knee rehabilitation |
| Ex6 | Full Body | Combined movements |

### Data Format
- **Joint Coordinates**: NumPy files (.npy) with shape (frames, joints, coordinates)
- **Joint Count**: 26 joints per skeleton frame
- **Segmentation**: CSV file with exercise metadata (video_id, repetition, correctness label)

### Correctness Labels
- **1 = Correct**: Exercise performed according to clinical guidelines
- **0 = Incorrect**: Exercise performed with deviations/errors

### Sample Distribution (Segmentation.csv)
```
Columns: video_id, repetition_number, exercise_id, person_id, 
         first_frame, last_frame, cam17_orientation, 
         mocap_erroneous, exercise_subtype, correctness
```

---

# SLIDE 8: Proposed Enhancements/Novelty (1/3)

## Beyond the Research Paper

### Novel Contributions - Overview

The base research paper provides core methodology. Our implementation extends this with:

| Enhancement Area | Research Paper | Our Extension |
|------------------|----------------|---------------|
| Architecture | Monolithic | **Edge-Cloud Distributed** |
| Communication | Direct | **Publisher-Subscriber Model** |
| Missing Data | No handling | **Joint Imputation Module** |
| Error Detection | Basic | **Compensation Detection** |
| Temporal Analysis | Frame-by-frame | **LSTM Fluidity Analysis** |
| Data Management | Not specified | **NoSQL with XML Format** |

### Team Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT ENHANCEMENTS                        │
├─────────────────────────────┬───────────────────────────────────┤
│     SUBTEAM 1 (EDGE)        │      SUBTEAM 2 (CLOUD)            │
├─────────────────────────────┼───────────────────────────────────┤
│ • Joint Imputation          │ • Publisher-Subscriber Model      │
│ • Compensation Detection    │ • NoSQL Database + XML Format     │
│ • LSTM Fluidity Analysis    │ • REST API Development            │
│                             │ • Clinician Dashboard             │
│                             │ • Population Benchmarking         │
└─────────────────────────────┴───────────────────────────────────┘
```

---

# SLIDE 9: Proposed Enhancements/Novelty (2/3)

## Subteam 1 - Edge Client Enhancements

### 1. Joint Imputation Module
**Problem**: MoveNet may fail to detect certain joints due to occlusion, poor lighting, or body positioning

**Solution**: Intelligent imputation of missing keypoints

```
Methods:
├── Temporal Interpolation: Use previous/next frames
├── Spatial Inference: Infer from neighboring joints
└── Motion Prediction: Predict based on movement trajectory
```

**Implementation**:
- Detect missing joints (confidence < threshold)
- Apply interpolation using temporal context
- Maintain anatomical constraints (bone length preservation)

### 2. Compensation Detection
**Problem**: Patients may achieve correct final position using incorrect body mechanics (e.g., leaning trunk instead of raising arm)

**Solution**: Detect compensatory movement patterns

```
Compensatory Pattern Examples:
├── Trunk Lean: Lateral body shift during arm exercises
├── Shoulder Shrug: Elevated shoulder during reaching
└── Hip Shift: Pelvis rotation during leg exercises
```

**Detection Method**:
- Monitor auxiliary joints not directly involved in exercise
- Flag unexpected movement in stabilizer joints
- Provide specific feedback on compensation type

### 3. LSTM Fluidity Analysis (End-Semester)
**Purpose**: Assess smoothness and naturalness of movement over time

**Metrics**:
- Movement jerkiness (third derivative of position)
- Velocity consistency
- Acceleration patterns

---

# SLIDE 10: Proposed Enhancements/Novelty (3/3)

## Subteam 2 - Cloud Server Enhancements

### 1. Publisher-Subscriber Communication Model
**Design**: Decoupled message-based architecture using MQTT/ZeroMQ

```
┌─────────────┐     Publish      ┌─────────────┐     Subscribe    ┌─────────────┐
│  Edge       │ ───────────────→ │   Message   │ ───────────────→ │   Cloud     │
│  Client     │   Joint Data     │   Broker    │   Joint Data     │   Server    │
└─────────────┘                  └─────────────┘                  └─────────────┘
```

**Advantages**:
- Scalable: Multiple edge clients, single cloud server
- Resilient: Survives temporary network disconnections
- Extensible: Easy to add new subscribers

### 2. NoSQL Database with XML Format
**Purpose**: Flexible schema for heterogeneous rehabilitation data

**Data Structure**:
```xml
<session>
  <user_id>patient_001</user_id>
  <timestamp>2025-02-17T10:30:00</timestamp>
  <exercise>Ex1</exercise>
  <frames>
    <frame id="1">
      <joints>
        <joint name="left_shoulder" x="0.45" y="0.32" conf="0.95"/>
        ...
      </joints>
    </frame>
  </frames>
  <correctness>1</correctness>
  <confidence>0.87</confidence>
</session>
```

### 3. REST API & Clinician Dashboard
- **API Endpoints**: Session upload, history retrieval, analytics queries
- **Dashboard**: Progress visualization, patient comparison, trend analysis

---

# SLIDE 11: Work Done - Current Status (1/2)

## Mid-Semester Implementation Progress (Modules 1-5)

### Module 1: MoveNet Pose Estimation ✓ Completed
**File**: `subteam1_edge/movenet.py`

**Implementation**:
- TensorFlow Hub integration for MoveNet Thunder model
- 17 keypoint extraction with (x, y, confidence)
- Video frame processing pipeline
- Fallback mode for CPU-only systems

**Key Code Features**:
```python
class MoveNetPoseEstimator:
    - load_model(): Initialize TensorFlow Hub model
    - process_frame(): Extract keypoints from single frame
    - process_video(): Batch process entire video
    - visualize_skeleton(): Draw skeleton on frame
```

### Module 2: Skeleton Normalization ✓ Completed
**File**: `subteam1_edge/normalization_17.py`

**Implementation**:
- Hip center calculation (midpoint of left/right hip)
- Translation to origin
- Torso length scaling (hip to shoulder distance)
- Scale-invariant coordinates output

**Normalization Formula**:
$$P_{normalized} = \frac{P_{original} - Hip_{center}}{Torso_{length}}$$

### Module 3: Reference Model Building ✓ Completed
**File**: `subteam1_edge/reference_model.py`

**Implementation**:
- Load correct execution samples from dataset
- Compute mean joint positions across correct samples
- Calculate standard deviation for variability bounds
- Store reference template for comparison

---

# SLIDE 12: Work Done - Current Status (2/2)

## Mid-Semester Implementation Progress (Continued)

### Module 4: Movement Comparison ✓ Completed
**File**: `subteam1_edge/comparison.py`

**Implementation**:
- Euclidean distance computation between observed and reference
- Per-joint distance analysis for localized feedback
- MSE and RMSE calculation
- Movement deviation scoring

**Metrics Implemented**:
| Metric | Formula | Purpose |
|--------|---------|---------|
| Euclidean Distance | $C = \|\|Y_{ref} - Y_{obs}\|\|$ | Overall deviation |
| Per-Joint Distance | $d_i = \sqrt{(x_i - x'_i)^2 + (y_i - y'_i)^2}$ | Identify problem joints |
| MSE | $\frac{1}{N}\sum(y_i - y'_i)^2$ | Average error |
| RMSE | $\sqrt{MSE}$ | Interpretable error |

### Module 5: Confidence Scoring & Classification ✓ Completed
**Files**: `subteam1_edge/confidence.py`, `subteam1_edge/classifier.py`

**Implementation**:
- Confidence scoring from RMSE values
- 0.8 threshold for reliable predictions
- Binary classification (correct = 1, incorrect = 0)
- Warning system for low-confidence predictions

**Classification Pipeline**:
```
Input → MoveNet → Normalize → Compare → Score → Classify → Output
 │                                                        │
 └── Video Frame                         Correct/Incorrect ──┘
```

### Integration Demo
**File**: `main.py`
- End-to-end pipeline execution
- Command-line interface for exercise selection
- Evaluation mode with accuracy reporting

---

# SLIDE 13: Individual Contribution (1/2)

## Subteam 1 - Edge Client (2 Members)

### Member 1: Pose Estimation & Normalization

| Task | Status | Description |
|------|--------|-------------|
| MoveNet Integration | ✓ Complete | TensorFlow Hub model loading and inference |
| Frame Processing | ✓ Complete | Video frame extraction and preprocessing |
| 17 Keypoint Extraction | ✓ Complete | Extract (x, y, confidence) for each joint |
| Skeleton Visualization | ✓ Complete | Draw skeleton overlay on frames |
| Hip Center Calculation | ✓ Complete | Compute midpoint of hip joints |
| Torso Length Scaling | ✓ Complete | Normalize by torso length |
| Translation to Origin | ✓ Complete | Center skeleton at hip |

**Files Developed**:
- `subteam1_edge/movenet.py` (351 lines)
- `subteam1_edge/normalization_17.py` (277 lines)

### Member 2: Reference Model & Comparison

| Task | Status | Description |
|------|--------|-------------|
| Dataset Loading | ✓ Complete | Load .npy files from REHAB24-6 |
| Reference Template | ✓ Complete | Compute mean from correct samples |
| Euclidean Distance | ✓ Complete | Implement deviation calculation |
| MSE/RMSE Metrics | ✓ Complete | Error quantification |
| Reference Statistics | ✓ Complete | Compute std for variability bounds |

**Files Developed**:
- `subteam1_edge/reference_model.py`
- `subteam1_edge/comparison.py` (270 lines)

---

# SLIDE 14: Individual Contribution (2/2)

## Subteam 2 - Cloud Server (2 Members)

### Member 3: Confidence Scoring & Classification

| Task | Status | Description |
|------|--------|-------------|
| Confidence Formula | ✓ Complete | Implement 1/(1+RMSE) scoring |
| Threshold Logic | ✓ Complete | 0.8 cutoff for reliable predictions |
| Warning System | ✓ Complete | Flag low-confidence detections |
| Classification Pipeline | ✓ Complete | Binary correct/incorrect output |
| Pipeline Integration | ✓ Complete | Connect all modules in sequence |
| Evaluation Mode | ✓ Complete | Accuracy computation on test set |

**Files Developed**:
- `subteam1_edge/confidence.py` (247 lines)
- `subteam1_edge/classifier.py` (411 lines)
- `main.py` (integration script)

### Member 4: Data Management & API Foundation

| Task | Status | Description |
|------|--------|-------------|
| Database Schema | ✓ Complete | SQLite session storage design |
| Session Storage | ✓ Complete | Store exercise sessions |
| REST API Setup | ✓ Complete | FastAPI framework configuration |
| API Endpoints | ✓ Complete | Session upload, query endpoints |
| Documentation | ✓ Complete | API docs via Swagger/OpenAPI |

**Files Developed**:
- `subteam2_cloud/database.py`
- `subteam2_cloud/api.py`
- `requirements.txt`
- `README.md`

---

## Summary Table: Mid-Semester Deliverables

| Module | Component | Owner | Status |
|--------|-----------|-------|--------|
| 1 | MoveNet Pose Estimation | Member 1 | ✓ Complete |
| 2 | Skeleton Normalization | Member 1 | ✓ Complete |
| 3 | Reference Model Building | Member 2 | ✓ Complete |
| 4 | Movement Comparison | Member 2 | ✓ Complete |
| 5 | Confidence & Classification | Member 3 | ✓ Complete |
| - | Database & API Foundation | Member 4 | ✓ Complete |

---

# END OF MID-SEMESTER CONTENT

## Coming in End-Semester:
- Joint Imputation Module
- Compensation Detection
- LSTM Fluidity Analysis
- Publisher-Subscriber Implementation
- NoSQL + XML Storage
- Clinician Dashboard
- Population Benchmarking
