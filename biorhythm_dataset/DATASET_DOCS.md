# BioRhythm Fusion Band — Training Dataset Documentation

## Overview

A structured, synthetic yet physiologically-realistic training dataset for deep learning-based **multivariate anomaly detection** in wearable health monitoring. The dataset is designed for the **BioRhythm Fusion Band** system and supports LSTM, Transformer, and hybrid model architectures.

---

## Dataset Files

| File | Description | Size (approx.) |
|---|---|---|
| `biorhythm_raw.csv` | Full continuous time-series rows | ~2.5 GB |
| `biorhythm_windows.npz` | Sliding-window tensors ready for model training | ~800 MB |
| `biorhythm_metadata.json` | Schema, statistics, class distribution, normalization params | ~5 KB |

---

## Signal Channels (7 Features)

| # | Signal | Unit | Range | Physiological Meaning |
|---|---|---|---|---|
| 1 | **HR** | bpm | 40–220 | Heart Rate |
| 2 | **HRV** | ms (RMSSD) | 5–130 | Heart Rate Variability — autonomic balance proxy |
| 3 | **SKT** | °C | 30–40 | Skin Temperature — inflammation / vasomotor indicator |
| 4 | **EDA** | μS | 0.1–20 | Electrodermal Activity — sympathetic nervous system load |
| 5 | **SPO2** | % | 88–100 | Peripheral Blood Oxygen Saturation (PPG-derived) |
| 6 | **SMF** | 0–1 | 0–1 | Sleep Micro-Fragmentation Index |
| 7 | **CRS** | 0–1 | 0–1 | Circadian Rhythm Stability Score |

---

## Disease Risk Class Labels (8 Classes)

| Label | Class Name | Key Signal Pattern |
|---|---|---|
| **0** | Normal Baseline | All signals within personal baseline ±σ |
| **1** | Early Viral Infection | HR↑, HRV↓, SKT↑, EDA↑, SMF↑ — pre-fever signature |
| **2** | Chronic Inflammation | Sustained HRV↓, high EDA, elevated SKT |
| **3** | Overtraining / Physical Burnout | HRV↓↓, HR↑, high EDA, disrupted sleep |
| **4** | Hormonal Imbalance | SKT volatility↑, SMF↑↑, CRS↓ |
| **5** | Early Metabolic Disorder | HRV↓, mild HR↑, SPO2 drift |
| **6** | Immune Suppression | HRV↓, blunted EDA, slight SKT↓ |
| **7** | Psychological Burnout | EDA sustained high, HRV↓↓, CRS↓↓, SMF↑ |

> **Key Design Principle**: Individual signals remain near-normal values in all disease classes. The model must learn to detect **correlated multi-signal micro-deviations** — not threshold breaches.

---

## Dataset Structure

```
biorhythm_dataset/
├── generate_dataset.py      # Generator script
├── biorhythm_raw.csv        # Full time-series
├── biorhythm_windows.npz    # Windowed tensors (NPZ)
│     ├── X_train  (N_train, 60, 7)  float32
│     ├── y_train  (N_train,)         int8
│     ├── X_val    (N_val,   60, 7)  float32
│     ├── y_val    (N_val,)           int8
│     ├── X_test   (N_test,  60, 7)  float32
│     ├── y_test   (N_test,)          int8
│     ├── feat_mean (7,)
│     ├── feat_std  (7,)
│     └── feature_names (7,)
└── biorhythm_metadata.json  # Full schema + statistics
```

### Raw CSV Schema

```
subject_id, timestamp, age, gender, HR, HRV, SKT, EDA, SPO2, SMF, CRS, label, label_name
0, 2026-01-01 00:00:00, 34, F, 72.41, 48.23, 34.512, 3.241, 97.82, 0.0841, 0.8812, 0, Normal_Baseline
...
```

---

## Dataset Statistics

| Property | Value |
|---|---|
| Subjects | 50 |
| Days per subject | 14 |
| Total raw samples | ~60 million |
| Window size | 60 seconds |
| Window stride | 30 seconds (50% overlap) |
| Total windows | ~40 million |
| Train / Val / Test | 70% / 15% / 15% |
| Normalization | Z-score per feature |

---

## Simulation Methodology

### 1. Subject-Level Personalization
Each of the 50 subjects gets a unique physiological baseline drawn from realistic population distributions (e.g., resting HR 62–82 bpm, HRV 28–68 ms).

### 2. Circadian Rhythm Modeling
All signals incorporate a 24-hour sinusoidal modulation with physiologically-correct phases:
- HR peaks around 2–4 PM, troughs 4–6 AM
- SKT inversely cycles (warmer at night)
- CRS scores dip during late-night fragmentation

```python
circadian(t) = A · sin(2π·t/86400 + φ)
```

### 3. 1/f (Pink) Noise
Each signal uses colored noise (power spectral density ∝ 1/f) to replicate the natural autocorrelation structure of physiological time-series — unlike white noise, this avoids unrealistically rapid oscillations.

### 4. Disease Episode Injection
Per subject, 2–5 disease episodes (30 min – 2 hrs each) are randomly placed. Each episode applies class-specific multi-signal perturbations with randomized severity (0.5–1.2×).

### 5. Hard Physiological Clipping
All values are clipped to physically impossible extremes to prevent artifact data.

---

## How to Load and Use

```python
import numpy as np

# Load windowed dataset
data = np.load("biorhythm_windows.npz")
X_train, y_train = data["X_train"], data["y_train"]  # (N, 60, 7), (N,)
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
# → Train: (28000000, 60, 7), Labels: (28000000,)
```

---

## Recommended Model Architectures

### Option A — LSTM Autoencoder (Unsupervised Anomaly Detection)
```python
# Input: (batch, 60, 7)  →  Encoder  →  Latent  →  Decoder  →  Reconstruction
# Anomaly score = MSE(input, reconstruction)
# Threshold on reconstruction error to detect anomalies
```

### Option B — Bidirectional LSTM Classifier (Supervised)
```python
Input(60, 7) → BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Dense(32) → Softmax(8)
```

### Option C — Transformer Encoder Classifier
```python
Input(60, 7) → PosEncoding → 4×TransformerBlock(heads=4, d=64) → GlobalAvgPool → Dense(8)
```

### Option D — CNN-LSTM Hybrid (Best for local + temporal)
```python
Input(60, 7) → Conv1D(64, k=3) → MaxPool → LSTM(128) → Dense(64) → Softmax(8)
```

---

## Training Tips

- **Class imbalance**: Use weighted cross-entropy or SMOTE (label 0 = ~45% of data)
- **Evaluation metrics**: F1-macro, AUROC per class, Sensitivity/Specificity
- **Baseline to beat**: Rule-based threshold system (F1 ≈ 0.42 on this dataset)
- **Target F1**: ≥ 0.85 macro on test set

---

## Extending the Dataset

To add more subjects or days, edit `CONFIG` in `generate_dataset.py`:
```python
CONFIG["num_subjects"]   = 200   # scale up
CONFIG["days_per_subject"] = 30  # longer monitoring
```
