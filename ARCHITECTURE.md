# System Architecture & Workflow Diagram

## 🏗️ Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  DEEPFAKE VOICE DETECTION SYSTEM                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌─────────────────┐      ┌──────────────┐
│   DATA LAYER    │      │  PROCESSING     │      │   MODEL      │
│                 │      │     LAYER       │      │    LAYER     │
├─────────────────┤      ├─────────────────┤      ├──────────────┤
│                 │      │                 │      │              │
│ • Real Voices   │─────▶│ • Preprocessing │─────▶│ • CNN Model  │
│ • Fake Voices   │      │ • Feature Ext.  │      │ • Training   │
│ • Test Audio    │      │ • Normalization │      │ • Inference  │
│                 │      │                 │      │              │
└─────────────────┘      └─────────────────┘      └──────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐              ┌──────────────────┐       │
│  │  Streamlit Web   │              │   CLI Tool       │       │
│  │  Application     │              │  (Batch/Single)  │       │
│  └──────────────────┘              └──────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Processing Pipeline

```
INPUT AUDIO FILE
      │
      ▼
┌──────────────────────────────────────┐
│  1. LOAD AUDIO (preprocess.py)       │
│     • Read file with librosa         │
│     • Resample to 16 kHz            │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  2. PREPROCESS (preprocess.py)       │
│     • Normalize to [-1, 1]          │
│     • Trim silence (optional)       │
│     • Pad/truncate to 3 seconds     │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  3. EXTRACT FEATURES (features.py)   │
│     • Mel Spectrogram (128 bands)   │
│     • OR MFCC (40 coefficients)     │
│     • Convert to (H x W x 1)        │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  4. NORMALIZE FEATURES (features.py) │
│     • Scale to [0, 1]               │
│     • Pad to consistent width       │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  5. CNN INFERENCE (model.py)         │
│     • Forward pass through model    │
│     • Get probability [0, 1]        │
└──────────────────────────────────────┘
      │
      ▼
PREDICTION: REAL (0) or FAKE (1)
```

---

## 🧠 CNN Model Architecture

```
INPUT: Mel Spectrogram (128 x Time x 1)
      │
      ▼
┌─────────────────────────────────────────┐
│  BLOCK 1: Convolutional                 │
│  • Conv2D(32, 3x3, ReLU, padding=same)  │
│  • BatchNormalization                   │
│  • MaxPooling2D(2x2)                   │
│  • Dropout(0.25)                        │
└─────────────────────────────────────────┘
      │ → Output: (64 x Time/2 x 32)
      ▼
┌─────────────────────────────────────────┐
│  BLOCK 2: Convolutional                 │
│  • Conv2D(64, 3x3, ReLU, padding=same)  │
│  • BatchNormalization                   │
│  • MaxPooling2D(2x2)                   │
│  • Dropout(0.25)                        │
└─────────────────────────────────────────┘
      │ → Output: (32 x Time/4 x 64)
      ▼
┌─────────────────────────────────────────┐
│  BLOCK 3: Convolutional                 │
│  • Conv2D(128, 3x3, ReLU, padding=same) │
│  • BatchNormalization                   │
│  • MaxPooling2D(2x2)                   │
│  • Dropout(0.3)                         │
└─────────────────────────────────────────┘
      │ → Output: (16 x Time/8 x 128)
      ▼
┌─────────────────────────────────────────┐
│  BLOCK 4: Convolutional                 │
│  • Conv2D(256, 3x3, ReLU, padding=same) │
│  • BatchNormalization                   │
│  • MaxPooling2D(2x2)                   │
│  • Dropout(0.3)                         │
└─────────────────────────────────────────┘
      │ → Output: (8 x Time/16 x 256)
      ▼
┌─────────────────────────────────────────┐
│  FLATTEN                                │
│  • Flatten to 1D vector                 │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  DENSE LAYERS                           │
│  • Dense(256, ReLU)                     │
│  • BatchNormalization                   │
│  • Dropout(0.5)                         │
│  • Dense(128, ReLU)                     │
│  • BatchNormalization                   │
│  • Dropout(0.5)                         │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  OUTPUT LAYER                           │
│  • Dense(1, Sigmoid)                    │
└─────────────────────────────────────────┘
      │
      ▼
OUTPUT: Probability P(Fake) ∈ [0, 1]
  • P < 0.5 → REAL
  • P ≥ 0.5 → FAKE
```

---

## 📊 Training Workflow

```
START
  │
  ▼
┌────────────────────────────────┐
│ Load Dataset                   │
│ • data/real/*.wav             │
│ • data/fake/*.wav             │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Preprocess All Files          │
│ • Resample, normalize, trim   │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Extract Features              │
│ • Mel Spectrograms            │
│ • Pad to consistent size      │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Split Dataset                 │
│ • 80% Training               │
│ • 20% Testing                │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Create Model                  │
│ • Build CNN architecture      │
│ • Compile with Adam optimizer │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Train Model                   │
│ • Epochs: 50                 │
│ • Batch size: 32             │
│ • Callbacks:                 │
│   - Early Stopping           │
│   - LR Reduction             │
│   - Model Checkpoint         │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Evaluate on Test Set          │
│ • Calculate metrics          │
│ • Generate plots             │
└────────────────────────────────┘
  │
  ▼
┌────────────────────────────────┐
│ Save Model                    │
│ • models/deepfake_detector.h5│
└────────────────────────────────┘
  │
  ▼
END
```

---

## 🌐 Streamlit App Workflow

```
USER OPENS WEB APP (localhost:8501)
      │
      ▼
┌──────────────────────────────────┐
│  DISPLAY INTERFACE               │
│  • Header & Description          │
│  • Upload Widget                 │
│  • Settings Sidebar              │
└──────────────────────────────────┘
      │
      ▼
USER UPLOADS AUDIO FILE
      │
      ▼
┌──────────────────────────────────┐
│  LOAD MODEL                      │
│  • Load deepfake_detector.h5     │
└──────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  PROCESS AUDIO                   │
│  • Preprocess uploaded file      │
│  • Extract features              │
│  • Normalize & pad               │
└──────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  MAKE PREDICTION                 │
│  • Run model.predict()           │
│  • Calculate confidence          │
└──────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  DISPLAY RESULTS                 │
│  ┌────────────────────────────┐ │
│  │ • Waveform Plot            │ │
│  │ • Spectrogram Plot         │ │
│  │ • Prediction Box           │ │
│  │   - REAL or FAKE           │ │
│  │   - Confidence %           │ │
│  │ • Detailed Metrics         │ │
│  │ • Audio Player             │ │
│  └────────────────────────────┘ │
└──────────────────────────────────┘
      │
      ▼
USER CAN UPLOAD ANOTHER FILE
```

---

## 🔄 Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                         config.py                           │
│                  (Configuration Settings)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  preprocess.py   │  │   features.py    │  │    model.py      │
│                  │  │                  │  │                  │
│ • load_audio     │  │ • extract_mel    │  │ • create_cnn     │
│ • normalize      │  │ • extract_mfcc   │  │ • compile        │
│ • trim_silence   │  │ • prepare_input  │  │ • save/load      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      train.py        │
                    │  (Training Script)   │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ deepfake_detector.h5 │
                    │   (Trained Model)    │
                    └──────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐
         │     app.py       │  │ example_usage.py │
         │  (Streamlit UI)  │  │   (CLI Tool)     │
         └──────────────────┘  └──────────────────┘
```

---

## 📁 File System Flow

```
PROJECT ROOT
│
├── 📄 Configuration Files
│   ├── config.py          → Centralized settings
│   ├── requirements.txt   → Python dependencies
│   └── setup.py          → Package installer
│
├── 🔧 Executable Scripts
│   ├── train.py          → Run to train model
│   ├── app.py            → Run to start web app
│   ├── example_usage.py  → Run for CLI predictions
│   └── test_setup.py     → Run to verify setup
│
├── 📦 Source Code (src/)
│   ├── preprocess.py     → Audio preprocessing
│   ├── features.py       → Feature extraction
│   └── model.py          → Model architecture
│
├── 💾 Data (data/)
│   ├── real/             → Real voice samples
│   └── fake/             → Fake voice samples
│
├── 🤖 Models (models/)
│   └── deepfake_detector.h5  → Trained model (created)
│
└── 📊 Outputs (plots/)
    ├── training_history.png   → Training curves
    ├── confusion_matrix.png   → Classification matrix
    └── roc_curve.png          → ROC curve
```

---

## ⚡ Quick Command Reference

```bash
# Setup
pip install -r requirements.txt
python test_setup.py

# Training
python train.py

# Web Application
streamlit run app.py

# CLI Prediction
python example_usage.py

# Configuration
python config.py

# Package Installation
pip install -e .
```

---

## 🎯 Key Metrics Flow

```
AUDIO INPUT
      │
      ▼
FEATURES (Mel Spectrogram)
      │
      ▼
CNN MODEL
      │
      ▼
RAW PROBABILITY P ∈ [0, 1]
      │
      ├─────────────┬─────────────┐
      │             │             │
      ▼             ▼             ▼
  P < 0.5       P = 0.5       P > 0.5
   REAL        UNCERTAIN       FAKE
      │             │             │
      └─────────────┴─────────────┘
                    │
                    ▼
         CONFIDENCE SCORE
              │
              ├─── Confidence = P (if FAKE)
              └─── Confidence = 1-P (if REAL)
                         │
                         ▼
              ┌──────────────────────┐
              │  High: ≥ 70%         │
              │  Medium: 50-70%      │
              │  Low: < 50%          │
              └──────────────────────┘
```

---

**Diagram Version**: 1.0  
**Last Updated**: February 2026
