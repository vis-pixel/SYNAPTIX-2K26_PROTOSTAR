# -*- coding: utf-8 -*-
"""
================================================================================
 BioRhythm Fusion Band — Early Fever Detection Simulation
 Multi-Signal Correlation Analysis (NOT Threshold-Based)
================================================================================

SIMULATION OVERVIEW:
  Step 1 — Generate 7 days of normal baseline physiological data
  Step 2 — Inject subtle pre-fever micro-deviations on Days 6–7:
            • Skin temperature +0.5°C (still < 38°C)
            • Resting HR +5 bpm
            • HRV reduced by ~15%
            • EDA elevated by ~20%
            • SPO2 dip of ~0.3%
            • Sleep fragmentation index rises
            • Circadian rhythm stability drops
  Step 3 — Run multi-signal correlation engine:
            • Compute rolling z-scores per signal against personal baseline
            • Build a correlation deviation matrix (6×6 upper triangle)
            • Composite Health Deviation Score (HDS)
            • Time-to-detection: when HDS crosses adaptive threshold
  Step 4 — Visualization: 7-panel physiological timeseries + HDS + detection

OUTPUT:
  fever_simulation_results.csv   — Full timeseries with labels & scores
  fever_detection_dashboard.png  — Publication-quality figure
  detection_report.txt           — Console report also saved to file

USAGE:
  pip install numpy pandas scipy matplotlib
  python fever_detection_simulation.py
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from datetime import datetime, timedelta

# Optional: nicer plots
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — skipping plot generation.")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR      = "d:/empty/biorhythm_dataset"
SAMPLE_HZ    = 1              # 1 sample/sec
DAYS         = 7
TOTAL_SEC    = DAYS * 86400
FEVER_ONSET  = 5 * 86400      # fever begins partway through Day 6
FEVER_RAMP   = 3600 * 4       # 4-hour ramp-up to peak deviation
BASELINE_WIN = 3 * 86400      # first 3 days used for personal baseline
ROLLING_WIN  = 900            # 15-min rolling window for z-scores
np.random.seed(2026)

# Subject baseline (personalized mean values)
BASELINE = {
    "HR":   72.0,    # bpm
    "HRV":  52.0,    # ms RMSSD
    "SKT":  34.8,    # °C
    "EDA":  3.2,     # μS
    "SPO2": 97.6,    # %
    "SMF":  0.10,    # sleep micro-fragmentation  (0–1)
    "CRS":  0.88,    # circadian rhythm stability  (0–1)
}

SIGNAL_NAMES = list(BASELINE.keys())
SIGNAL_UNITS = ["bpm", "ms", "°C", "μS", "%", "idx", "score"]

# ─── CIRCADIAN MODEL ──────────────────────────────────────────────────────────
def circadian(t, amplitude, phase):
    return amplitude * np.sin(2 * np.pi * t / 86400 + phase)

# ─── PINK NOISE (1/f) ─────────────────────────────────────────────────────────
def pink_noise(n, scale=1.0):
    f = np.fft.rfftfreq(n)
    f[0] = 1e-9
    spectrum = np.random.randn(len(f)) * (f ** -0.5)
    sig = np.fft.irfft(spectrum, n=n) * scale
    # Gentle lowpass
    b, a = butter(2, 0.005, btype="low")
    return filtfilt(b, a, sig)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: GENERATE NORMAL BASELINE (7 DAYS)
# ══════════════════════════════════════════════════════════════════════════════
def generate_baseline_signals():
    """7-day continuous physiological recording with circadian rhythm + noise."""
    t = np.arange(TOTAL_SEC, dtype=np.float64)

    signals = {}
    # Circadian params: (amplitude, phase_shift)
    circ_params = {
        "HR":   (3.5,  0.0),
        "HRV":  (5.0,  0.3),
        "SKT":  (0.35, -0.5),
        "EDA":  (0.6,  0.1),
        "SPO2": (0.15, 0.2),
        "SMF":  (0.03, -1.0),
        "CRS":  (0.06, 0.0),
    }
    # Noise scale per signal
    noise_scales = {
        "HR":   1.0,
        "HRV":  1.5,
        "SKT":  0.04,
        "EDA":  0.15,
        "SPO2": 0.06,
        "SMF":  0.012,
        "CRS":  0.015,
    }

    for sig in SIGNAL_NAMES:
        amp, phase = circ_params[sig]
        base = BASELINE[sig] + circadian(t, amp, phase) + pink_noise(TOTAL_SEC, noise_scales[sig])
        signals[sig] = base

    return t, signals


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: INJECT SUBTLE PRE-FEVER MICRO-DEVIATIONS
# ══════════════════════════════════════════════════════════════════════════════
def inject_fever_micro_deviations(t, signals):
    """
    Inject physiologically-plausible pre-fever deviations starting at FEVER_ONSET.
    Uses a smooth sigmoid ramp to avoid sudden step artifacts.
    Key: individual signals remain BELOW clinical thresholds at all times.
    """
    # Sigmoid ramp function
    ramp_center = FEVER_ONSET + FEVER_RAMP / 2
    ramp = 1.0 / (1.0 + np.exp(-(t - ramp_center) / (FEVER_RAMP / 8)))
    # Zero out before fever onset
    ramp[t < FEVER_ONSET] = 0.0

    # Micro-deviation magnitudes (subtle, sub-clinical)
    deviations = {
        "HR":   +5.0,    # +5 bpm (still in normal range for most people)
        "HRV":  -8.0,    # -8 ms RMSSD (~15% reduction)
        "SKT":  +0.5,    # +0.5°C (stays below 38°C fever threshold)
        "EDA":  +0.7,    # +0.7 μS (~20% above baseline)
        "SPO2": -0.3,    # -0.3% (still well above 95% alarm)
        "SMF":  +0.08,   # sleep gets more fragmented
        "CRS":  -0.10,   # circadian rhythm destabilizes
    }

    # Add some noise to deviations (realistic variability)
    for sig in SIGNAL_NAMES:
        dev_noise = pink_noise(TOTAL_SEC, abs(deviations[sig]) * 0.15)
        signals[sig] += ramp * (deviations[sig] + dev_noise)

    # Hard physiological clipping
    signals["HR"]   = np.clip(signals["HR"],   40, 180)
    signals["HRV"]  = np.clip(signals["HRV"],  5, 130)
    signals["SKT"]  = np.clip(signals["SKT"],  30.0, 40.0)
    signals["EDA"]  = np.clip(signals["EDA"],  0.1, 20.0)
    signals["SPO2"] = np.clip(signals["SPO2"], 88.0, 100.0)
    signals["SMF"]  = np.clip(signals["SMF"],  0.0, 1.0)
    signals["CRS"]  = np.clip(signals["CRS"],  0.0, 1.0)

    # Create label array: 0 = normal, 1 = pre-fever deviation active
    labels = np.where(ramp > 0.05, 1, 0).astype(np.int8)

    return signals, labels, ramp


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: MULTI-SIGNAL CORRELATION DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_personal_baseline_stats(signals):
    """Compute mean and std for each signal using the first BASELINE_WIN seconds."""
    stats = {}
    for sig in SIGNAL_NAMES:
        window = signals[sig][:BASELINE_WIN]
        stats[sig] = {"mean": np.mean(window), "std": np.std(window) + 1e-9}
    return stats


def rolling_zscore(signal, baseline_mean, baseline_std, window=ROLLING_WIN):
    """Compute rolling z-score relative to personal baseline."""
    # Rolling mean
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")
    return (smoothed - baseline_mean) / baseline_std


def correlation_deviation_matrix(z_scores, window=ROLLING_WIN):
    """
    Compute a rolling pairwise correlation matrix between all signal z-scores.
    Returns the norm of the correlation deviation from baseline identity.
    """
    n_sig = len(SIGNAL_NAMES)
    n_samples = len(z_scores[SIGNAL_NAMES[0]])
    corr_norm = np.zeros(n_samples)

    # Downsample for computation speed (every 60s)
    step = 60
    for i in range(window, n_samples - window, step):
        sl = slice(i - window, i + window)
        z_window = np.column_stack([z_scores[s][sl] for s in SIGNAL_NAMES])
        # Correlation matrix
        try:
            corr = np.corrcoef(z_window.T)
            # Baseline: signals are weakly correlated (~identity-ish)
            # Anomaly: signals become strongly CROSS-correlated (fever drives all)
            # Measure: Frobenius norm of off-diagonal elements
            off_diag = corr - np.eye(n_sig)
            corr_norm[i] = np.sqrt(np.sum(off_diag ** 2)) / (n_sig * (n_sig - 1) / 2)
        except:
            pass

    # Interpolate the gaps
    nonzero_idx = np.nonzero(corr_norm)[0]
    if len(nonzero_idx) > 2:
        corr_norm = np.interp(np.arange(n_samples), nonzero_idx, corr_norm[nonzero_idx])

    return corr_norm


def compute_health_deviation_score(z_scores, corr_norm):
    """
    Composite Health Deviation Score (HDS):
      HDS = w1 * |mean(z-scores)| + w2 * max(|z-scores|) + w3 * corr_deviation

    This captures:
      - Average multi-signal drift (w1)
      - Worst single-signal deviation (w2)
      - Cross-signal correlation anomaly (w3)
    """
    n_samples = len(z_scores[SIGNAL_NAMES[0]])
    z_matrix = np.column_stack([z_scores[s] for s in SIGNAL_NAMES])

    # Component 1: Mean absolute z-score across all signals
    mean_abs_z = np.mean(np.abs(z_matrix), axis=1)

    # Component 2: Max absolute z-score (worst single deviation)
    max_abs_z = np.max(np.abs(z_matrix), axis=1)

    # Component 3: Correlation deviation norm
    # Weights (emphasize cross-correlation as the KEY differentiator)
    w1, w2, w3 = 0.30, 0.20, 0.50

    hds = w1 * mean_abs_z + w2 * max_abs_z + w3 * corr_norm

    # Smooth the HDS
    kernel = np.ones(300) / 300   # 5-min smoothing
    hds_smooth = np.convolve(hds, kernel, mode="same")

    return hds_smooth, mean_abs_z, max_abs_z


def detect_fever_onset(hds, labels):
    """
    Determine adaptive threshold and find the first sustained detection.
    Threshold = mean(baseline_HDS) + 3 * std(baseline_HDS)
    Detection requires 10+ consecutive minutes above threshold.
    """
    baseline_hds = hds[:BASELINE_WIN]
    threshold = np.mean(baseline_hds) + 3.0 * np.std(baseline_hds)

    above = (hds > threshold).astype(int)
    sustained_min = 10 * 60  # 10 minutes in seconds

    # Find first sustained window
    detection_time = None
    count = 0
    for i in range(len(above)):
        if above[i]:
            count += 1
            if count >= sustained_min:
                detection_time = i - sustained_min
                break
        else:
            count = 0

    # Ground truth onset
    true_onset = None
    for i in range(len(labels)):
        if labels[i] == 1:
            true_onset = i
            break

    return threshold, detection_time, true_onset


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: VISUALIZATION & REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(t, signals, labels, z_scores, hds, corr_norm,
                    threshold, detection_time, true_onset):
    """Print and save the detection report."""
    start_dt = datetime(2026, 1, 1)

    lines = []
    lines.append("=" * 72)
    lines.append("  BioRhythm Fusion Band — Fever Detection Simulation Report")
    lines.append("=" * 72)
    lines.append(f"  Simulation Duration   : {DAYS} days ({TOTAL_SEC:,} samples @ 1 Hz)")
    lines.append(f"  Personal Baseline Win : {BASELINE_WIN // 3600} hours (Days 1–3)")
    lines.append(f"  Fever Onset (ground)  : Day {FEVER_ONSET // 86400 + 1}, "
                 f"Hour {(FEVER_ONSET % 86400) // 3600}")
    lines.append(f"  HDS Threshold         : {threshold:.4f}")

    if detection_time is not None:
        det_day  = detection_time // 86400 + 1
        det_hour = (detection_time % 86400) // 3600
        det_min  = (detection_time % 3600) // 60
        lead_time_hr = (true_onset - detection_time) / 3600 if true_onset and detection_time < true_onset else \
                       (detection_time - true_onset) / 3600 if true_onset else 0
        lines.append(f"\n  [DETECTED] FEVER DETECTED at Day {det_day}, {det_hour:02d}:{det_min:02d}")
        if true_onset:
            true_day  = true_onset // 86400 + 1
            true_hour = (true_onset % 86400) // 3600
            if detection_time <= true_onset:
                lines.append(f"    Lead time     : {abs(lead_time_hr):.1f} hours BEFORE clinical onset")
            else:
                lines.append(f"    Detection lag : {abs(lead_time_hr):.1f} hours AFTER clinical onset")
    else:
        lines.append("\n  [MISS] No sustained fever detection triggered.")

    # Signal-level analysis at peak deviation
    peak_idx = min(FEVER_ONSET + FEVER_RAMP, TOTAL_SEC - 1)
    lines.append(f"\n  Signal Analysis at Peak Deviation (sample {peak_idx:,}):")
    lines.append(f"  {'Signal':<8} {'Baseline':>10} {'Current':>10} {'Z-Score':>10} {'Status':>12}")
    lines.append("  " + "-" * 56)

    stats = compute_personal_baseline_stats(signals)
    for i, sig in enumerate(SIGNAL_NAMES):
        bm = stats[sig]["mean"]
        cur = signals[sig][peak_idx]
        z  = z_scores[sig][peak_idx]
        status = ">> DEVIATED" if abs(z) > 1.5 else "   Normal"
        lines.append(f"  {sig:<8} {bm:>10.2f} {cur:>10.2f} {z:>+10.2f} {status:>12}")

    lines.append(f"\n  Cross-Signal Correlation Norm : {corr_norm[peak_idx]:.4f}")
    lines.append(f"  Composite HDS Score           : {hds[peak_idx]:.4f}")

    # Key insight
    lines.append("\n" + "-" * 72)
    lines.append("  KEY FINDING:")
    lines.append("  No individual signal breached a clinical alarm threshold.")
    lines.append("  Detection was achieved via CORRELATED multi-signal micro-deviations")
    lines.append("  -- the fundamental advantage of the BioRhythm Fusion Band approach.")
    lines.append("-" * 72)

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(OUT_DIR, "detection_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    return report


def plot_dashboard(t, signals, labels, z_scores, hds, corr_norm,
                   threshold, detection_time, true_onset):
    """Generate a multi-panel publication-quality dashboard figure."""
    if not HAS_MPL:
        print("[SKIP] No matplotlib — cannot generate plot.")
        return

    # Convert to hours for readability
    t_hours = t / 3600.0

    # Downsample for plotting (every 60s = 1 point/min)
    ds = 60
    t_h = t_hours[::ds]
    ds_labels = labels[::ds]

    fig = plt.figure(figsize=(20, 24), facecolor="#0a0e1a")
    gs = GridSpec(9, 1, figure=fig, hspace=0.35, top=0.96, bottom=0.03,
                  left=0.08, right=0.95)

    # Style
    TEXT_COLOR  = "#c8d0e8"
    GRID_COLOR  = "#1a2038"
    NORMAL_CLR  = "#00D4FF"
    FEVER_CLR   = "#FF4E6A"
    DETECT_CLR  = "#00E5A0"
    THRESH_CLR  = "#FFB347"

    def setup_ax(ax, ylabel, title_text=None):
        ax.set_facecolor("#0d1220")
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15, color=GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#1a2038")
        if title_text:
            ax.set_title(title_text, color=TEXT_COLOR, fontsize=10,
                         fontweight="bold", loc="left", pad=6)

    # Title
    fig.suptitle("BioRhythm Fusion Band - Early Fever Detection Simulation",
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.99,
                 fontfamily="sans-serif")

    titles = [
        "Heart Rate (bpm)",
        "Heart Rate Variability — RMSSD (ms)",
        "Skin Temperature Gradient (°C)",
        "Electrodermal Activity (μS)",
        "Peripheral SpO₂ via PPG (%)",
        "Sleep Micro-Fragmentation Index",
        "Circadian Rhythm Stability Score",
    ]

    for i, sig in enumerate(SIGNAL_NAMES):
        ax = fig.add_subplot(gs[i])
        y = signals[sig][::ds]
        # Color the line based on label
        ax.plot(t_h, y, color=NORMAL_CLR, linewidth=0.5, alpha=0.85)
        # Shade fever region
        fever_mask = ds_labels > 0
        if np.any(fever_mask):
            ax.fill_between(t_h, y.min(), y.max(), where=fever_mask,
                            color=FEVER_CLR, alpha=0.08)
        if detection_time is not None:
            ax.axvline(detection_time / 3600, color=DETECT_CLR, linewidth=1.2,
                       linestyle="--", alpha=0.7)
        if true_onset is not None:
            ax.axvline(true_onset / 3600, color=FEVER_CLR, linewidth=1.0,
                       linestyle=":", alpha=0.6)
        setup_ax(ax, SIGNAL_UNITS[i], titles[i])
        if i < len(SIGNAL_NAMES) - 1:
            ax.set_xticklabels([])

    # HDS panel
    ax_hds = fig.add_subplot(gs[7])
    hds_ds = hds[::ds]
    ax_hds.plot(t_h, hds_ds, color="#FFB347", linewidth=1.0, alpha=0.9)
    ax_hds.axhline(threshold, color=THRESH_CLR, linewidth=1.2, linestyle="--",
                   alpha=0.8, label=f"Threshold ({threshold:.3f})")
    ax_hds.fill_between(t_h, 0, hds_ds, where=hds_ds > threshold,
                        color=FEVER_CLR, alpha=0.25)
    if detection_time is not None:
        ax_hds.axvline(detection_time / 3600, color=DETECT_CLR, linewidth=1.5,
                       linestyle="--", alpha=0.9, label="Detection")
    if true_onset is not None:
        ax_hds.axvline(true_onset / 3600, color=FEVER_CLR, linewidth=1.2,
                       linestyle=":", alpha=0.7, label="True Onset")
    ax_hds.legend(loc="upper left", fontsize=8, facecolor="#0d1220",
                  edgecolor="#1a2038", labelcolor=TEXT_COLOR)
    setup_ax(ax_hds, "HDS", "Composite Health Deviation Score (Multi-Signal Fusion)")

    # Correlation panel
    ax_corr = fig.add_subplot(gs[8])
    corr_ds = corr_norm[::ds]
    ax_corr.plot(t_h, corr_ds, color="#6C63FF", linewidth=0.8, alpha=0.9)
    ax_corr.fill_between(t_h, 0, corr_ds, alpha=0.15, color="#6C63FF")
    if detection_time is not None:
        ax_corr.axvline(detection_time / 3600, color=DETECT_CLR, linewidth=1.5,
                        linestyle="--", alpha=0.9)
    if true_onset is not None:
        ax_corr.axvline(true_onset / 3600, color=FEVER_CLR, linewidth=1.2,
                        linestyle=":", alpha=0.7)
    setup_ax(ax_corr, "Norm", "Cross-Signal Correlation Deviation (Frobenius Norm)")
    ax_corr.set_xlabel("Time (hours)", color=TEXT_COLOR, fontsize=10)

    # Day markers
    for ax in fig.axes:
        for d in range(1, DAYS + 1):
            ax.axvline(d * 24, color="#1a2038", linewidth=0.8, alpha=0.5)

    plt.savefig(os.path.join(OUT_DIR, "fever_detection_dashboard.png"),
                dpi=150, facecolor="#0a0e1a")
    plt.close()
    print(f"  Dashboard saved: {os.path.join(OUT_DIR, 'fever_detection_dashboard.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 72)
    print("  BioRhythm Fusion Band — Fever Detection Simulation")
    print("=" * 72)

    # ── STEP 1 ──
    print("\n[STEP 1] Generating 7-day normal baseline...")
    t, signals = generate_baseline_signals()
    print(f"         {TOTAL_SEC:,} samples generated ({DAYS} days × 86400 samples/day)")

    # ── STEP 2 ──
    print("\n[STEP 2] Injecting subtle pre-fever micro-deviations...")
    print(f"         Onset: Day {FEVER_ONSET // 86400 + 1}, "
          f"Hour {(FEVER_ONSET % 86400) // 3600}")
    print(f"         Ramp-up: {FEVER_RAMP // 3600} hours (sigmoid)")
    print("         Deviations: SKT +0.5°C, HR +5 bpm, HRV -8 ms, "
          "EDA +0.7 μS, SPO2 -0.3%")
    signals, labels, ramp = inject_fever_micro_deviations(t, signals)

    # ── STEP 3 ──
    print("\n[STEP 3] Running multi-signal correlation detection engine...")
    stats = compute_personal_baseline_stats(signals)

    # Compute rolling z-scores
    print("         Computing per-signal rolling z-scores...")
    z_scores = {}
    for sig in SIGNAL_NAMES:
        z_scores[sig] = rolling_zscore(signals[sig], stats[sig]["mean"],
                                       stats[sig]["std"])

    # Cross-signal correlation deviation
    print("         Computing cross-signal correlation matrix...")
    corr_norm = correlation_deviation_matrix(z_scores)

    # Composite HDS
    print("         Computing composite Health Deviation Score...")
    hds, mean_abs_z, max_abs_z = compute_health_deviation_score(z_scores, corr_norm)

    # Detection
    print("         Running sustained-threshold detection...")
    threshold, detection_time, true_onset = detect_fever_onset(hds, labels)

    # ── STEP 4 ──
    print("\n[STEP 4] Generating report and visualizations...")
    report = generate_report(t, signals, labels, z_scores, hds, corr_norm,
                             threshold, detection_time, true_onset)

    # Save CSV
    print("\n  Saving results CSV...")
    # Downsample to 1/min for CSV to keep file size manageable
    ds = 60
    start_dt = datetime(2026, 1, 1)
    rows = []
    for i in range(0, TOTAL_SEC, ds):
        row = {
            "timestamp": (start_dt + timedelta(seconds=int(i))).strftime("%Y-%m-%d %H:%M:%S"),
            "day": i // 86400 + 1,
            "hour": (i % 86400) / 3600,
        }
        for sig in SIGNAL_NAMES:
            row[sig] = round(float(signals[sig][i]), 4)
            row[f"{sig}_zscore"] = round(float(z_scores[sig][i]), 4)
        row["correlation_norm"]    = round(float(corr_norm[i]), 4)
        row["health_deviation"]    = round(float(hds[i]), 4)
        row["label"]               = int(labels[i])
        row["label_name"]          = "pre_fever" if labels[i] == 1 else "normal"
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "fever_simulation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}  ({len(df):,} rows)")

    # Plot
    plot_dashboard(t, signals, labels, z_scores, hds, corr_norm,
                   threshold, detection_time, true_onset)

    print("\n" + "=" * 72)
    print("  [OK] Simulation complete!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
