"""
================================================================================
 BioRhythm Fusion Band — Structured Training Dataset Generator
 For Multivariate Anomaly Detection (LSTM / Transformer / Hybrid Models)
================================================================================

Signals Simulated (7 channels):
  1. HR      — Heart Rate (bpm)
  2. HRV     — Heart Rate Variability (ms, RMSSD)
  3. SKT     — Skin Temperature (°C)
  4. EDA     — Electrodermal Activity (μS)
  5. SPO2    — Peripheral SpO2 via PPG (%)
  6. SMF     — Sleep Micro-Fragmentation Index (0–1)
  7. CRS     — Circadian Rhythm Stability Score (0–1)

Disease Risk Labels (multi-class):
  0 — Normal / Healthy Baseline
  1 — Early Viral Infection Signature
  2 — Chronic Inflammation Risk
  3 — Overtraining / Physical Burnout
  4 — Hormonal Imbalance Pattern
  5 — Early Metabolic Disorder
  6 — Immune Suppression Window
  7 — Long-Term Burnout / Psychological Stress

Output Files:
  biorhythm_raw.csv          — Full continuous time-series (per-sample rows)
  biorhythm_windows.npz      — Sliding-window tensors (N, T, F) ready for LSTM
  biorhythm_metadata.json    — Dataset statistics, schema, and class distribution

Usage:
  pip install numpy pandas scipy tqdm
  python generate_dataset.py
================================================================================
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "num_subjects":         50,          # Number of simulated patients
    "days_per_subject":     14,          # Days of continuous monitoring per subject
    "sample_rate_hz":       1,           # 1 sample per second (downsampled from 250 Hz)
    "window_size":          60,          # 60-second sliding window for model input
    "window_stride":        30,          # 50% overlap stride
    "output_dir":           "d:/empty/biorhythm_dataset",
    "random_seed":          42,
    "signals":              ["HR", "HRV", "SKT", "EDA", "SPO2", "SMF", "CRS"],
    "class_names": {
        0: "Normal_Baseline",
        1: "Early_Viral_Infection",
        2: "Chronic_Inflammation",
        3: "Overtraining_Burnout",
        4: "Hormonal_Imbalance",
        5: "Early_Metabolic_Disorder",
        6: "Immune_Suppression",
        7: "Psychological_Burnout"
    },
    # Approximate probability of each class in the dataset
    "class_distribution": [0.45, 0.12, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05]
}

np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─── PHYSIOLOGICAL BASELINE RANGES (per subject variability) ──────────────────
def generate_subject_baseline():
    """Generate a unique personalized physiological baseline for one subject."""
    return {
        "HR_mean":   np.random.uniform(62, 82),      # bpm
        "HRV_mean":  np.random.uniform(28, 68),      # ms (RMSSD)
        "SKT_mean":  np.random.uniform(33.2, 36.5),  # °C
        "EDA_mean":  np.random.uniform(1.5, 6.0),    # μS
        "SPO2_mean": np.random.uniform(96.5, 99.0),  # %
        "SMF_mean":  np.random.uniform(0.05, 0.20),  # fragmentation index
        "CRS_mean":  np.random.uniform(0.75, 0.95),  # circadian stability
        "gender":    random.choice(["M", "F"]),
        "age":       np.random.randint(22, 72),
    }

# ─── CIRCADIAN RHYTHM MODEL ───────────────────────────────────────────────────
def circadian_modulation(t_seconds, amplitude=1.0, phase_shift=0.0):
    """
    Sinusoidal circadian modulation (24-hour cycle).
    Peak around 2–4 PM (36000s), trough around 4–6 AM (18000s).
    """
    period = 86400  # seconds in 24 hours
    phase  = 2 * np.pi * (t_seconds / period) + phase_shift
    return amplitude * np.sin(phase - np.pi / 2)   # normalized to [-1, 1]

# ─── LOWPASS FILTER for smoothness ───────────────────────────────────────────
def smooth_signal(signal, cutoff=0.01, fs=1.0, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)

# ─── NOISE GENERATOR ─────────────────────────────────────────────────────────
def colored_noise(n, alpha=1.0):
    """Generate 1/f^alpha (pink) noise for realistic physiological variation."""
    f  = np.fft.rfftfreq(n)
    f[0] = 1e-6  # avoid division by zero
    spectrum = np.random.randn(len(f)) * (f ** (-alpha / 2))
    return np.fft.irfft(spectrum, n=n)

# ─── DISEASE PERTURBATION FUNCTIONS ──────────────────────────────────────────
def apply_disease_perturbation(signals, label, severity=1.0):
    """
    Apply physiologically-plausible multi-signal micro-deviations for each
    disease class. Individual signals stay near-normal; the PATTERN is the anomaly.
    """
    HR, HRV, SKT, EDA, SPO2, SMF, CRS = signals

    if label == 1:  # Early Viral Infection (48h pre-fever signature)
        HR   += severity * np.random.uniform(3, 7)
        HRV  -= severity * np.random.uniform(4, 9)
        SKT  += severity * np.random.uniform(0.2, 0.6)
        EDA  += severity * np.random.uniform(0.3, 0.8)
        SPO2 -= severity * np.random.uniform(0.1, 0.4)
        SMF  += severity * np.random.uniform(0.05, 0.15)
        CRS  -= severity * np.random.uniform(0.05, 0.12)

    elif label == 2:  # Chronic Inflammation
        HR   += severity * np.random.uniform(2, 5)
        HRV  -= severity * np.random.uniform(6, 14)
        SKT  += severity * np.random.uniform(0.15, 0.4)
        EDA  += severity * np.random.uniform(0.5, 1.2)
        SPO2 -= severity * np.random.uniform(0.05, 0.3)
        SMF  += severity * np.random.uniform(0.08, 0.18)
        CRS  -= severity * np.random.uniform(0.08, 0.15)

    elif label == 3:  # Overtraining / Physical Burnout
        HR   += severity * np.random.uniform(5, 12)
        HRV  -= severity * np.random.uniform(10, 22)
        SKT  += severity * np.random.uniform(0.1, 0.35)
        EDA  += severity * np.random.uniform(1.0, 2.5)
        SPO2 -= severity * np.random.uniform(0.2, 0.6)
        SMF  += severity * np.random.uniform(0.10, 0.25)
        CRS  -= severity * np.random.uniform(0.08, 0.18)

    elif label == 4:  # Hormonal Imbalance
        HR   += severity * np.random.uniform(1, 4)
        HRV  -= severity * np.random.uniform(3, 7)
        SKT  += severity * np.random.uniform(0.3, 0.8)   # hot flashes / dysregulation
        EDA  += severity * np.random.uniform(0.6, 1.4)
        SPO2 -= severity * np.random.uniform(0.0, 0.2)
        SMF  += severity * np.random.uniform(0.12, 0.28)  # disrupted sleep
        CRS  -= severity * np.random.uniform(0.12, 0.22)

    elif label == 5:  # Early Metabolic Disorder
        HR   += severity * np.random.uniform(2, 6)
        HRV  -= severity * np.random.uniform(5, 11)
        SKT  += severity * np.random.uniform(0.05, 0.25)
        EDA  += severity * np.random.uniform(0.2, 0.7)
        SPO2 -= severity * np.random.uniform(0.1, 0.35)
        SMF  += severity * np.random.uniform(0.06, 0.14)
        CRS  -= severity * np.random.uniform(0.06, 0.13)

    elif label == 6:  # Immune Suppression Window
        HR   += severity * np.random.uniform(1, 4)
        HRV  -= severity * np.random.uniform(7, 15)
        SKT  -= severity * np.random.uniform(0.1, 0.4)   # slightly LOWER (suppression)
        EDA  -= severity * np.random.uniform(0.1, 0.5)   # blunted response
        SPO2 -= severity * np.random.uniform(0.05, 0.25)
        SMF  += severity * np.random.uniform(0.04, 0.12)
        CRS  -= severity * np.random.uniform(0.10, 0.20)

    elif label == 7:  # Long-term Burnout / Psychological Stress
        HR   += severity * np.random.uniform(3, 8)
        HRV  -= severity * np.random.uniform(8, 18)
        SKT  += severity * np.random.uniform(0.05, 0.22)
        EDA  += severity * np.random.uniform(1.2, 3.0)   # sustained high EDA
        SPO2 -= severity * np.random.uniform(0.1, 0.3)
        SMF  += severity * np.random.uniform(0.15, 0.30)
        CRS  -= severity * np.random.uniform(0.15, 0.25)

    return HR, HRV, SKT, EDA, SPO2, SMF, CRS

# ─── GENERATE ONE SUBJECT'S TIME SERIES ──────────────────────────────────────
def generate_subject_data(subject_id, baseline):
    """
    Produce N seconds of continuous multivariate physiological time-series
    for a single subject over 14 days.
    """
    N = CONFIG["days_per_subject"] * 86400   # total samples
    rows = []

    # Build circadian time axis
    t = np.arange(N)

    # Per-signal circadian modulation (each signal has slightly different phase)
    circ_HR   = circadian_modulation(t, amplitude=4.0,  phase_shift=0.0)
    circ_HRV  = circadian_modulation(t, amplitude=6.0,  phase_shift=0.3)
    circ_SKT  = circadian_modulation(t, amplitude=0.4,  phase_shift=-0.5)
    circ_EDA  = circadian_modulation(t, amplitude=0.8,  phase_shift=0.1)
    circ_SPO2 = circadian_modulation(t, amplitude=0.2,  phase_shift=0.2)
    circ_SMF  = circadian_modulation(t, amplitude=0.04, phase_shift=-1.0)
    circ_CRS  = circadian_modulation(t, amplitude=0.08, phase_shift=0.0)

    # Colored (pink) noise per signal for realistic variation
    noise_scale = [1.2, 2.0, 0.06, 0.2, 0.08, 0.02, 0.02]
    noises = [smooth_signal(colored_noise(N) * s, cutoff=0.005) for s in noise_scale]

    # Assign disease labels over time segments (simulate episodes)
    labels = np.zeros(N, dtype=np.int8)
    class_dist = CONFIG["class_distribution"]

    # Inject 2–5 disease episodes of different types per subject
    num_episodes = np.random.randint(2, 6)
    for _ in range(num_episodes):
        ep_class    = np.random.choice(range(1, 8), p=np.array(class_dist[1:]) / sum(class_dist[1:]))
        ep_start    = np.random.randint(0, N - 7200)
        ep_duration = np.random.randint(1800, 7200)    # 30min–2hr episode
        labels[ep_start:ep_start + ep_duration] = ep_class

    # Build sample-by-sample rows
    start_time = datetime(2026, 1, 1, 0, 0, 0) + timedelta(days=subject_id)
    chunk_size  = 3600  # process 1 hour at a time for memory efficiency

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        sl = slice(chunk_start, chunk_end)

        HR   = baseline["HR_mean"]   + circ_HR[sl]   + noises[0][sl]
        HRV  = baseline["HRV_mean"]  + circ_HRV[sl]  + noises[1][sl]
        SKT  = baseline["SKT_mean"]  + circ_SKT[sl]  + noises[2][sl]
        EDA  = baseline["EDA_mean"]  + circ_EDA[sl]  + noises[3][sl]
        SPO2 = baseline["SPO2_mean"] + circ_SPO2[sl] + noises[4][sl]
        SMF  = baseline["SMF_mean"]  + circ_SMF[sl]  + noises[5][sl]
        CRS  = baseline["CRS_mean"]  + circ_CRS[sl]  + noises[6][sl]

        chunk_labels = labels[sl]

        # Apply disease perturbations
        for i in range(len(HR)):
            lbl = chunk_labels[i]
            if lbl > 0:
                severity = np.random.uniform(0.5, 1.2)
                HR[i], HRV[i], SKT[i], EDA[i], SPO2[i], SMF[i], CRS[i] = \
                    apply_disease_perturbation(
                        (HR[i], HRV[i], SKT[i], EDA[i], SPO2[i], SMF[i], CRS[i]),
                        lbl, severity
                    )

        # Hard physiological clipping
        HR   = np.clip(HR,   40,  220)
        HRV  = np.clip(HRV,  5,   130)
        SKT  = np.clip(SKT,  30.0, 40.0)
        EDA  = np.clip(EDA,  0.1,  20.0)
        SPO2 = np.clip(SPO2, 88.0, 100.0)
        SMF  = np.clip(SMF,  0.0,  1.0)
        CRS  = np.clip(CRS,  0.0,  1.0)

        for i in range(len(HR)):
            ts = start_time + timedelta(seconds=chunk_start + i)
            rows.append({
                "subject_id":  subject_id,
                "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                "age":         baseline["age"],
                "gender":      baseline["gender"],
                "HR":          round(float(HR[i]), 2),
                "HRV":         round(float(HRV[i]), 2),
                "SKT":         round(float(SKT[i]), 3),
                "EDA":         round(float(EDA[i]), 3),
                "SPO2":        round(float(SPO2[i]), 2),
                "SMF":         round(float(SMF[i]), 4),
                "CRS":         round(float(CRS[i]), 4),
                "label":       int(chunk_labels[i]),
                "label_name":  CONFIG["class_names"][int(chunk_labels[i])],
            })

    return rows, labels

# ─── SLIDING WINDOW EXTRACTION ────────────────────────────────────────────────
def extract_windows(df):
    """
    Extract overlapping sliding windows from the full time-series dataframe.
    Returns:
        X: ndarray (N_windows, window_size, n_features)
        y: ndarray (N_windows,)  — majority label in window
        meta: list of dicts with subject_id and window start time
    """
    features = CONFIG["signals"]
    W  = CONFIG["window_size"]
    S  = CONFIG["window_stride"]
    X_list, y_list, meta_list = [], [], []

    subjects = df["subject_id"].unique()
    iterator = tqdm(subjects, desc="Extracting windows") if HAS_TQDM else subjects

    for sid in iterator:
        sub_df = df[df["subject_id"] == sid].reset_index(drop=True)
        n = len(sub_df)
        for start in range(0, n - W, S):
            window_df = sub_df.iloc[start:start + W]
            X_window  = window_df[features].values.astype(np.float32)
            # Majority label in the window
            majority_label = int(np.bincount(window_df["label"].values.astype(int)).argmax())
            X_list.append(X_window)
            y_list.append(majority_label)
            meta_list.append({
                "subject_id": int(sid),
                "window_start": window_df["timestamp"].iloc[0]
            })

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int8)
    return X, y, meta_list

# ─── Z-SCORE NORMALIZATION ────────────────────────────────────────────────────
def normalize_windows(X):
    """Per-feature z-score normalization across all windows."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std, mean.squeeze(), std.squeeze()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  BioRhythm Fusion Band — Training Dataset Generator")
    print("="*72)
    print(f"  Subjects       : {CONFIG['num_subjects']}")
    print(f"  Days/subject   : {CONFIG['days_per_subject']}")
    print(f"  Window size    : {CONFIG['window_size']}s")
    print(f"  Window stride  : {CONFIG['window_stride']}s")
    print(f"  Output dir     : {CONFIG['output_dir']}")
    print("="*72 + "\n")

    all_rows  = []
    all_labels = []
    t0 = time.time()

    iterator = range(CONFIG["num_subjects"])
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Generating subjects")

    for sid in iterator:
        baseline = generate_subject_baseline()
        rows, labels = generate_subject_data(sid, baseline)
        all_rows.extend(rows)
        all_labels.extend(labels)
        if not HAS_TQDM:
            print(f"  Subject {sid+1:02d}/{CONFIG['num_subjects']} done.")

    # ── SAVE RAW CSV ──
    print("\n[1/4] Saving raw time-series CSV...")
    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(CONFIG["output_dir"], "biorhythm_raw.csv")
    # Save in chunks to avoid memory issues
    chunk = 200_000
    for i, start in enumerate(range(0, len(df), chunk)):
        mode = "w" if i == 0 else "a"
        header = (i == 0)
        df.iloc[start:start+chunk].to_csv(raw_path, mode=mode, header=header, index=False)
    print(f"    Saved: {raw_path}  ({len(df):,} rows)")

    # ── EXTRACT WINDOWS ──
    print("\n[2/4] Extracting sliding windows...")
    X, y, meta = extract_windows(df)
    print(f"    Windows shape: X={X.shape}, y={y.shape}")

    # ── NORMALIZE ──
    print("\n[3/4] Normalizing features (z-score)...")
    X_norm, feat_mean, feat_std = normalize_windows(X)

    # Train / Val / Test split (70 / 15 / 15)
    n_total = len(X_norm)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    idx     = np.random.permutation(n_total)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    # ── SAVE .NPZ ──
    npz_path = os.path.join(CONFIG["output_dir"], "biorhythm_windows.npz")
    print(f"\n[4/4] Saving windowed tensors to .npz ...")
    np.savez_compressed(
        npz_path,
        X_train=X_norm[train_idx], y_train=y[train_idx],
        X_val=X_norm[val_idx],     y_val=y[val_idx],
        X_test=X_norm[test_idx],   y_test=y[test_idx],
        feat_mean=feat_mean,
        feat_std=feat_std,
        feature_names=np.array(CONFIG["signals"])
    )
    print(f"    Saved : {npz_path}")
    print(f"    Train : {len(train_idx):,} windows")
    print(f"    Val   : {len(val_idx):,} windows")
    print(f"    Test  : {len(test_idx):,} windows")

    # ── METADATA JSON ──
    label_arr = np.array(all_labels, dtype=int)
    class_counts = {CONFIG["class_names"][i]: int(np.sum(label_arr == i))
                    for i in range(8)}

    metadata = {
        "generated_at"   : datetime.now().isoformat(),
        "generator_ver"  : "1.0.0",
        "num_subjects"   : CONFIG["num_subjects"],
        "days_per_subject": CONFIG["days_per_subject"],
        "total_raw_samples": len(df),
        "total_windows"  : int(n_total),
        "train_windows"  : int(len(train_idx)),
        "val_windows"    : int(len(val_idx)),
        "test_windows"   : int(len(test_idx)),
        "window_size_sec": CONFIG["window_size"],
        "window_stride_sec": CONFIG["window_stride"],
        "signal_channels": CONFIG["signals"],
        "num_classes"    : 8,
        "class_names"    : CONFIG["class_names"],
        "class_distribution_samples": class_counts,
        "normalization"  : "z-score per feature",
        "feat_mean"      : feat_mean.tolist(),
        "feat_std"       : feat_std.tolist(),
        "schema": {
            "subject_id"  : "int — unique patient ID",
            "timestamp"   : "datetime — ISO-8601 UTC",
            "age"         : "int — subject age (22–72)",
            "gender"      : "str — M/F",
            "HR"          : "float — Heart Rate (bpm) [40–220]",
            "HRV"         : "float — HRV RMSSD (ms) [5–130]",
            "SKT"         : "float — Skin Temperature (°C) [30–40]",
            "EDA"         : "float — Electrodermal Activity (μS) [0.1–20]",
            "SPO2"        : "float — Peripheral SpO2 (%) [88–100]",
            "SMF"         : "float — Sleep Micro-Fragmentation Index [0–1]",
            "CRS"         : "float — Circadian Rhythm Stability Score [0–1]",
            "label"       : "int — Disease class [0–7]",
            "label_name"  : "str — Human-readable class name"
        },
        "model_input_shape": [CONFIG["window_size"], len(CONFIG["signals"])],
        "recommended_models": [
            "LSTM Autoencoder (anomaly detection via reconstruction error)",
            "Bidirectional LSTM + Dense classifier",
            "Transformer Encoder + MLP head",
            "CNN-LSTM Hybrid for local + temporal features",
            "Informer / TimesNet for long-range circadian dependencies"
        ]
    }

    meta_path = os.path.join(CONFIG["output_dir"], "biorhythm_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n    Metadata : {meta_path}")

    # ── SUMMARY ──
    elapsed = time.time() - t0
    print("\n" + "="*72)
    print("  ✓ Dataset generation complete!")
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"  Raw CSV        : {raw_path}")
    print(f"  Windowed .npz  : {npz_path}")
    print(f"  Metadata JSON  : {meta_path}")
    print("\n  Class distribution in raw samples:")
    for cls, count in class_counts.items():
        pct = 100.0 * count / len(label_arr)
        bar = "█" * int(pct / 2)
        print(f"    {cls:<30} {count:>8,}  ({pct:5.1f}%) {bar}")
    print("="*72 + "\n")

if __name__ == "__main__":
    main()
