"""
BioRhythm Fusion Band — Multi-Signal Training Dataset Generator
════════════════════════════════════════════════════════════════
Generates structured physiological time-series data for deep learning
models (LSTM / Transformer / hybrid) to detect early-stage disease risk
via multi-signal CORRELATION, not threshold-based alerts.

Signals: HR, HRV, SpO2, EDA, SkinTemp, Respiration, Steps, SleepStage, BP_sys, BP_dia
Labels:  healthy, pre_hypertension, early_diabetes, sleep_apnea, chronic_stress,
         early_infection, cardiac_risk, respiratory_risk

Run:
    python scripts/generate_training_dataset.py --subjects 500 --hours 72
    python scripts/generate_training_dataset.py --subjects 100 --hours 24 --quick
"""

import argparse
import csv
import json
import os
import random
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ─── Disease Profiles ─────────────────────────────────────────────────────────
# Each profile defines how vitals CORRELATE during early-stage disease
# The key insight: individual signals may be "normal", but CROSS-SIGNAL
# patterns reveal early risk — this is what the model must learn.

DISEASE_PROFILES = {
    "healthy": {
        "label_id": 0,
        "hr":           {"mean": 72,   "std": 8,   "drift": 0},
        "hrv_rmssd":    {"mean": 55,   "std": 10,  "drift": 0},
        "spo2":         {"mean": 97.5, "std": 0.8, "drift": 0},
        "eda":          {"mean": 2.0,  "std": 0.5, "drift": 0},
        "skin_temp":    {"mean": 33.5, "std": 0.3, "drift": 0},
        "respiration":  {"mean": 15,   "std": 2,   "drift": 0},
        "steps_per_hr": {"mean": 250,  "std": 150, "drift": 0},
        "bp_systolic":  {"mean": 118,  "std": 6,   "drift": 0},
        "bp_diastolic": {"mean": 76,   "std": 4,   "drift": 0},
        "sleep_quality":{"mean": 0.85, "std": 0.08,"drift": 0},
        "correlations": {
            "hr_hrv": -0.75,       # normal: HR ↑ → HRV ↓
            "hr_eda": 0.3,         # mild link
            "temp_hr": 0.15,       # weak
            "resp_spo2": -0.2,     # weak inverse
        },
    },
    "pre_hypertension": {
        "label_id": 1,
        "hr":           {"mean": 78,   "std": 10,  "drift": 0.08},  # slow uptrend
        "hrv_rmssd":    {"mean": 42,   "std": 12,  "drift": -0.05}, # declining
        "spo2":         {"mean": 96.8, "std": 1.0, "drift": 0},
        "eda":          {"mean": 3.2,  "std": 0.8, "drift": 0.02},  # rising stress
        "skin_temp":    {"mean": 33.8, "std": 0.4, "drift": 0},
        "respiration":  {"mean": 16,   "std": 2.5, "drift": 0},
        "steps_per_hr": {"mean": 180,  "std": 120, "drift": -0.3},  # declining activity
        "bp_systolic":  {"mean": 132,  "std": 8,   "drift": 0.12},  # gradual rise
        "bp_diastolic": {"mean": 85,   "std": 5,   "drift": 0.08},
        "sleep_quality":{"mean": 0.72, "std": 0.10,"drift": -0.01},
        "correlations": {
            "hr_hrv": -0.85,       # STRONGER inverse — key marker
            "hr_eda": 0.6,         # stress-HR coupling tightens
            "temp_hr": 0.4,        # temp tracks HR more
            "resp_spo2": -0.3,
        },
    },
    "early_diabetes": {
        "label_id": 2,
        "hr":           {"mean": 82,   "std": 12,  "drift": 0.04},
        "hrv_rmssd":    {"mean": 38,   "std": 14,  "drift": -0.08}, # autonomic dysfunction
        "spo2":         {"mean": 96.5, "std": 1.2, "drift": -0.01},
        "eda":          {"mean": 1.5,  "std": 0.6, "drift": -0.02}, # EDA DROPS (neuropathy)
        "skin_temp":    {"mean": 32.8, "std": 0.5, "drift": -0.02}, # cooler extremities
        "respiration":  {"mean": 17,   "std": 3,   "drift": 0.02},
        "steps_per_hr": {"mean": 140,  "std": 100, "drift": -0.5},  # fatigue
        "bp_systolic":  {"mean": 128,  "std": 9,   "drift": 0.06},
        "bp_diastolic": {"mean": 82,   "std": 5,   "drift": 0.04},
        "sleep_quality":{"mean": 0.68, "std": 0.12,"drift": -0.02},
        "correlations": {
            "hr_hrv": -0.6,        # WEAKER inverse — autonomic decoupling!
            "hr_eda": 0.15,        # EDA decouples from HR (neuropathy)
            "temp_hr": -0.3,       # INVERTED — cold extremities despite high HR
            "resp_spo2": -0.35,
        },
    },
    "sleep_apnea": {
        "label_id": 3,
        "hr":           {"mean": 75,   "std": 15,  "drift": 0},     # high variability
        "hrv_rmssd":    {"mean": 35,   "std": 18,  "drift": -0.03},
        "spo2":         {"mean": 94.5, "std": 3.0, "drift": 0},     # INTERMITTENT drops
        "eda":          {"mean": 2.8,  "std": 0.9, "drift": 0},
        "skin_temp":    {"mean": 33.2, "std": 0.6, "drift": 0},
        "respiration":  {"mean": 12,   "std": 5,   "drift": 0},     # IRREGULAR
        "steps_per_hr": {"mean": 160,  "std": 110, "drift": -0.2},
        "bp_systolic":  {"mean": 130,  "std": 10,  "drift": 0.05},
        "bp_diastolic": {"mean": 84,   "std": 6,   "drift": 0.03},
        "sleep_quality":{"mean": 0.55, "std": 0.15,"drift": -0.03},
        "correlations": {
            "hr_hrv": -0.5,
            "hr_eda": 0.5,
            "temp_hr": 0.1,
            "resp_spo2": -0.8,     # VERY STRONG — resp drops → SpO2 crash (apnea!)
        },
    },
    "chronic_stress": {
        "label_id": 4,
        "hr":           {"mean": 85,   "std": 12,  "drift": 0.06},
        "hrv_rmssd":    {"mean": 30,   "std": 10,  "drift": -0.1},  # very low HRV
        "spo2":         {"mean": 97.0, "std": 0.9, "drift": 0},
        "eda":          {"mean": 5.5,  "std": 1.5, "drift": 0.05},  # HIGH EDA
        "skin_temp":    {"mean": 34.2, "std": 0.5, "drift": 0.01},
        "respiration":  {"mean": 18,   "std": 3,   "drift": 0.03},
        "steps_per_hr": {"mean": 120,  "std": 90,  "drift": -0.4},
        "bp_systolic":  {"mean": 135,  "std": 10,  "drift": 0.1},
        "bp_diastolic": {"mean": 88,   "std": 6,   "drift": 0.06},
        "sleep_quality":{"mean": 0.58, "std": 0.14,"drift": -0.04},
        "correlations": {
            "hr_hrv": -0.9,        # very tight inverse
            "hr_eda": 0.8,         # VERY STRONG — EDA-HR locked (sympathetic)
            "temp_hr": 0.5,
            "resp_spo2": -0.25,
        },
    },
    "early_infection": {
        "label_id": 5,
        "hr":           {"mean": 88,   "std": 10,  "drift": 0.15},  # rising
        "hrv_rmssd":    {"mean": 36,   "std": 12,  "drift": -0.12},
        "spo2":         {"mean": 95.5, "std": 1.5, "drift": -0.05}, # slowly dropping
        "eda":          {"mean": 3.5,  "std": 1.0, "drift": 0.04},
        "skin_temp":    {"mean": 34.8, "std": 0.6, "drift": 0.08},  # RISING temp!
        "respiration":  {"mean": 19,   "std": 3,   "drift": 0.06},
        "steps_per_hr": {"mean": 100,  "std": 80,  "drift": -1.0},  # rapid decline
        "bp_systolic":  {"mean": 125,  "std": 8,   "drift": 0.05},
        "bp_diastolic": {"mean": 80,   "std": 5,   "drift": 0.03},
        "sleep_quality":{"mean": 0.62, "std": 0.15,"drift": -0.05},
        "correlations": {
            "hr_hrv": -0.8,
            "hr_eda": 0.55,
            "temp_hr": 0.75,       # STRONG — HR rises WITH temp (inflammatory)
            "resp_spo2": -0.6,     # resp effort can't maintain SpO2
        },
    },
    "cardiac_risk": {
        "label_id": 6,
        "hr":           {"mean": 80,   "std": 18,  "drift": 0.03},  # HIGH variability
        "hrv_rmssd":    {"mean": 25,   "std": 15,  "drift": -0.15}, # very low, dropping
        "spo2":         {"mean": 95.8, "std": 1.5, "drift": -0.02},
        "eda":          {"mean": 3.0,  "std": 0.8, "drift": 0.03},
        "skin_temp":    {"mean": 33.0, "std": 0.5, "drift": -0.01},
        "respiration":  {"mean": 17,   "std": 4,   "drift": 0.03},
        "steps_per_hr": {"mean": 100,  "std": 80,  "drift": -0.6},
        "bp_systolic":  {"mean": 145,  "std": 12,  "drift": 0.1},   # HIGH BP
        "bp_diastolic": {"mean": 92,   "std": 7,   "drift": 0.06},
        "sleep_quality":{"mean": 0.60, "std": 0.14,"drift": -0.03},
        "correlations": {
            "hr_hrv": -0.4,        # BROKEN inverse — cardiac dysfunction
            "hr_eda": 0.45,
            "temp_hr": -0.2,       # paradoxical: cool despite tachycardia
            "resp_spo2": -0.5,
        },
    },
    "respiratory_risk": {
        "label_id": 7,
        "hr":           {"mean": 80,   "std": 10,  "drift": 0.04},
        "hrv_rmssd":    {"mean": 40,   "std": 12,  "drift": -0.05},
        "spo2":         {"mean": 93.5, "std": 2.5, "drift": -0.08}, # key: LOW SpO2
        "eda":          {"mean": 2.8,  "std": 0.7, "drift": 0.02},
        "skin_temp":    {"mean": 34.0, "std": 0.5, "drift": 0.03},
        "respiration":  {"mean": 22,   "std": 4,   "drift": 0.08},  # ELEVATED resp rate
        "steps_per_hr": {"mean": 80,   "std": 60,  "drift": -0.8},
        "bp_systolic":  {"mean": 126,  "std": 8,   "drift": 0.04},
        "bp_diastolic": {"mean": 80,   "std": 5,   "drift": 0.02},
        "sleep_quality":{"mean": 0.55, "std": 0.16,"drift": -0.04},
        "correlations": {
            "hr_hrv": -0.65,
            "hr_eda": 0.4,
            "temp_hr": 0.35,
            "resp_spo2": -0.85,    # STRONGEST — resp can't compensate for SpO2
        },
    },
}

# ─── Circadian rhythm modulator ───────────────────────────────────────────────
def circadian_factor(hour):
    """Returns a multiplier based on 24h circadian rhythm."""
    # HR lowest ~4AM, highest ~2PM
    return 1.0 + 0.08 * math.sin(2 * math.pi * (hour - 4) / 24)

def activity_factor(hour):
    """Returns activity multiplier based on time of day."""
    # active: 8-12, 14-18; sleeping: 23-6
    if 23 <= hour or hour < 6:
        return 0.1
    elif 8 <= hour <= 12 or 14 <= hour <= 18:
        return 1.0 + random.uniform(0, 0.5)
    else:
        return 0.5 + random.uniform(0, 0.3)

def sleep_stage(hour):
    """Returns sleep stage: 0=awake, 1=light, 2=deep, 3=REM."""
    if 7 <= hour <= 22:
        return 0  # awake
    # Cycle: light → deep → REM (~ 90 min cycles)
    sleep_hour = (hour - 23) % 24 if hour >= 23 else hour + 1
    cycle_pos = (sleep_hour * 60) % 90  # position in cycle
    if cycle_pos < 25:
        return 1  # light
    elif cycle_pos < 50:
        return 2  # deep
    else:
        return 3  # REM

# ─── Correlated noise generator ───────────────────────────────────────────────
def correlated_noise(base1, base2, correlation, std1, std2):
    """Generate two correlated noise values using Cholesky-like approach."""
    z1 = random.gauss(0, 1)
    z2 = random.gauss(0, 1)
    n1 = z1
    n2 = correlation * z1 + math.sqrt(max(0, 1 - correlation**2)) * z2
    return base1 + n1 * std1, base2 + n2 * std2

# ─── Intermittent anomaly injector ────────────────────────────────────────────
def inject_apnea_event(spo2, respiration, hour, profile_name):
    """For sleep apnea: inject sudden SpO2 drops during sleep."""
    if profile_name == "sleep_apnea" and (23 <= hour or hour < 6):
        if random.random() < 0.15:  # 15% chance per sleep-hour reading
            spo2 -= random.uniform(4, 12)   # sudden desaturation
            respiration = max(2, respiration - random.uniform(5, 10))  # near-apnea
    return max(70, spo2), max(2, respiration)

# ─── Subject Generator ────────────────────────────────────────────────────────
def generate_subject(subject_id, profile_name, profile, hours, sample_rate_min=5):
    """Generate time-series for one subject with correlated multi-signal data."""
    rows = []
    samples_per_hour = 60 // sample_rate_min  # default: 12 readings/hour
    total_samples = hours * samples_per_hour

    # Subject demographics
    age = random.randint(22, 72)
    weight_kg = round(random.uniform(50, 110), 1)
    height_cm = random.randint(155, 190)
    gender = random.choice(["M", "F"])
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    # Individual baseline variation (±15% from profile mean)
    subject_offsets = {}
    for signal in ["hr", "hrv_rmssd", "spo2", "eda", "skin_temp", "respiration",
                    "steps_per_hr", "bp_systolic", "bp_diastolic", "sleep_quality"]:
        p = profile[signal]
        subject_offsets[signal] = random.gauss(0, p["std"] * 0.3)

    start_time = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 90))
    correlations = profile["correlations"]

    for i in range(total_samples):
        t = start_time + timedelta(minutes=i * sample_rate_min)
        hour = t.hour + t.minute / 60.0
        c_factor = circadian_factor(hour)
        a_factor = activity_factor(hour)
        s_stage = sleep_stage(int(hour))

        # Time-based drift (simulate disease progression)
        time_frac = i / max(1, total_samples - 1)

        # Base values with drift
        hr_base    = profile["hr"]["mean"]    + profile["hr"]["drift"]    * hours * time_frac + subject_offsets["hr"]
        hrv_base   = profile["hrv_rmssd"]["mean"] + profile["hrv_rmssd"]["drift"] * hours * time_frac + subject_offsets["hrv_rmssd"]
        spo2_base  = profile["spo2"]["mean"]  + profile["spo2"]["drift"]  * hours * time_frac + subject_offsets["spo2"]
        eda_base   = profile["eda"]["mean"]   + profile["eda"]["drift"]   * hours * time_frac + subject_offsets["eda"]
        temp_base  = profile["skin_temp"]["mean"] + profile["skin_temp"]["drift"] * hours * time_frac + subject_offsets["skin_temp"]
        resp_base  = profile["respiration"]["mean"] + profile["respiration"]["drift"] * hours * time_frac + subject_offsets["respiration"]
        steps_base = profile["steps_per_hr"]["mean"] + profile["steps_per_hr"]["drift"] * hours * time_frac + subject_offsets["steps_per_hr"]
        bp_s_base  = profile["bp_systolic"]["mean"] + profile["bp_systolic"]["drift"] * hours * time_frac + subject_offsets["bp_systolic"]
        bp_d_base  = profile["bp_diastolic"]["mean"] + profile["bp_diastolic"]["drift"] * hours * time_frac + subject_offsets["bp_diastolic"]
        slp_base   = profile["sleep_quality"]["mean"] + profile["sleep_quality"]["drift"] * hours * time_frac + subject_offsets["sleep_quality"]

        # Apply circadian + activity modulation
        hr_base  *= c_factor * (1 + 0.15 * (a_factor - 0.5))
        hrv_base *= (2 - c_factor)  # HRV inverse of HR circadian
        resp_base *= (0.8 + 0.2 * a_factor)
        steps_val = max(0, steps_base * a_factor)

        # Generate CORRELATED noise
        hr_val, hrv_val = correlated_noise(
            hr_base, hrv_base, correlations["hr_hrv"],
            profile["hr"]["std"], profile["hrv_rmssd"]["std"]
        )
        hr_val2, eda_val = correlated_noise(
            hr_val, eda_base, correlations["hr_eda"],
            0,  # don't add more noise to HR
            profile["eda"]["std"]
        )
        temp_val, _ = correlated_noise(
            temp_base, hr_val, correlations["temp_hr"],
            profile["skin_temp"]["std"], 0
        )
        resp_val, spo2_val = correlated_noise(
            resp_base, spo2_base, correlations["resp_spo2"],
            profile["respiration"]["std"], profile["spo2"]["std"]
        )

        # BP correlated with HR
        bp_s_val = bp_s_base + random.gauss(0, profile["bp_systolic"]["std"]) + (hr_val - 72) * 0.3
        bp_d_val = bp_d_base + random.gauss(0, profile["bp_diastolic"]["std"]) + (hr_val - 72) * 0.15

        # Sleep quality
        slp_val = max(0, min(1, slp_base + random.gauss(0, profile["sleep_quality"]["std"])))

        # Inject intermittent events (apnea)
        spo2_val, resp_val = inject_apnea_event(spo2_val, resp_val, int(hour), profile_name)

        # Clamp to physiological ranges
        hr_val   = max(35, min(200, round(hr_val, 1)))
        hrv_val  = max(5, min(150, round(hrv_val, 1)))
        spo2_val = max(70, min(100, round(spo2_val, 1)))
        eda_val  = max(0.1, round(eda_val, 2))
        temp_val = max(30, min(42, round(temp_val, 2)))
        resp_val = max(4, min(40, round(resp_val, 1)))
        steps_val= max(0, int(steps_val / samples_per_hour))  # steps per interval
        bp_s_val = max(80, min(200, round(bp_s_val, 0)))
        bp_d_val = max(50, min(130, round(bp_d_val, 0)))

        # Risk severity (0-100) based on signal deviations
        risk_score = compute_risk_score(profile_name, hr_val, hrv_val, spo2_val,
                                        eda_val, temp_val, resp_val, bp_s_val)

        rows.append({
            "timestamp":     t.isoformat(),
            "subject_id":    f"SUBJ-{subject_id:04d}",
            "age":           age,
            "gender":        gender,
            "bmi":           bmi,
            "heart_rate":    hr_val,
            "hrv_rmssd":     hrv_val,
            "spo2":          spo2_val,
            "eda":           eda_val,
            "skin_temp":     temp_val,
            "respiration":   resp_val,
            "steps":         steps_val,
            "bp_systolic":   int(bp_s_val),
            "bp_diastolic":  int(bp_d_val),
            "sleep_stage":   s_stage,
            "sleep_quality": round(slp_val, 3),
            "hour_of_day":   round(hour, 2),
            "time_fraction": round(time_frac, 4),
            "risk_score":    risk_score,
            "label":         profile_name,
            "label_id":      profile["label_id"],
        })

    return rows


def compute_risk_score(profile_name, hr, hrv, spo2, eda, temp, resp, bp_s):
    """Compute a 0-100 risk severity score based on deviation from healthy."""
    if profile_name == "healthy":
        return max(0, min(25, int(random.gauss(8, 5))))

    score = 0
    score += max(0, (hr - 80)) * 0.5
    score += max(0, (50 - hrv)) * 0.6
    score += max(0, (97 - spo2)) * 5
    score += max(0, (eda - 3)) * 3
    score += max(0, (temp - 34.5)) * 8
    score += max(0, (resp - 18)) * 2
    score += max(0, (bp_s - 130)) * 0.8
    return max(0, min(100, int(score + random.gauss(0, 4))))


# ─── Main Generator ──────────────────────────────────────────────────────────
def generate_dataset(n_subjects=500, hours=72, sample_rate_min=5, output_dir="datasets/training"):
    """Generate the full multi-subject, multi-condition dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Class distribution (slightly imbalanced — realistic)
    distribution = {
        "healthy":           0.30,
        "pre_hypertension":  0.12,
        "early_diabetes":    0.10,
        "sleep_apnea":       0.10,
        "chronic_stress":    0.12,
        "early_infection":   0.08,
        "cardiac_risk":      0.10,
        "respiratory_risk":  0.08,
    }

    all_rows = []
    subject_id = 0
    class_counts = {}

    print(f"\n{'='*60}")
    print(f"  BioRhythm Fusion Band — Training Dataset Generator")
    print(f"{'='*60}")
    print(f"  Subjects:      {n_subjects}")
    print(f"  Hours/subject: {hours}")
    print(f"  Sample rate:   every {sample_rate_min} min")
    print(f"  Samples/subj:  {hours * (60 // sample_rate_min)}")
    print(f"  Total samples: ~{n_subjects * hours * (60 // sample_rate_min):,}")
    print(f"  Classes:       {len(DISEASE_PROFILES)}")
    print(f"  Output:        {output_dir}/")
    print(f"{'='*60}\n")

    for condition, fraction in distribution.items():
        count = max(1, int(n_subjects * fraction))
        class_counts[condition] = count
        profile = DISEASE_PROFILES[condition]

        for j in range(count):
            subject_id += 1
            if subject_id > n_subjects:
                break

            rows = generate_subject(subject_id, condition, profile, hours, sample_rate_min)
            all_rows.extend(rows)

            # Progress
            pct = subject_id / n_subjects * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.0f}% | Subject {subject_id}/{n_subjects} ({condition})", end="", flush=True)

        if subject_id > n_subjects:
            break

    print(f"\n\n  ✓ Generated {len(all_rows):,} total samples for {subject_id} subjects\n")

    # ─── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "biorhythm_training_data.csv")
    fieldnames = list(all_rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  📄 CSV:  {csv_path} ({file_size_mb:.1f} MB)")

    # ─── Write metadata JSON ──────────────────────────────────────────────────
    metadata = {
        "dataset_name": "BioRhythm Fusion Band — Multi-Signal Training Data",
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "description": (
            "Synthetic multivariate physiological time-series for early-stage "
            "disease risk detection using cross-signal correlation patterns."
        ),
        "total_samples": len(all_rows),
        "total_subjects": subject_id,
        "hours_per_subject": hours,
        "sample_rate_minutes": sample_rate_min,
        "signals": [
            {"name": "heart_rate",    "unit": "BPM",   "range": [35, 200],  "description": "Heart rate"},
            {"name": "hrv_rmssd",     "unit": "ms",    "range": [5, 150],   "description": "HRV RMSSD"},
            {"name": "spo2",          "unit": "%",     "range": [70, 100],  "description": "Blood oxygen saturation"},
            {"name": "eda",           "unit": "μS",    "range": [0.1, 20],  "description": "Electrodermal activity"},
            {"name": "skin_temp",     "unit": "°C",    "range": [30, 42],   "description": "Peripheral skin temperature"},
            {"name": "respiration",   "unit": "BrPM",  "range": [4, 40],    "description": "Respiration rate"},
            {"name": "steps",         "unit": "count", "range": [0, 1000],  "description": "Steps per 5-min interval"},
            {"name": "bp_systolic",   "unit": "mmHg",  "range": [80, 200],  "description": "Systolic blood pressure"},
            {"name": "bp_diastolic",  "unit": "mmHg",  "range": [50, 130],  "description": "Diastolic blood pressure"},
            {"name": "sleep_stage",   "unit": "class",  "range": [0, 3],    "description": "0=awake, 1=light, 2=deep, 3=REM"},
            {"name": "sleep_quality", "unit": "score", "range": [0, 1],     "description": "Overall sleep quality index"},
        ],
        "labels": {
            name: {"id": p["label_id"], "count": class_counts.get(name, 0)}
            for name, p in DISEASE_PROFILES.items()
        },
        "class_distribution": class_counts,
        "key_correlations": {
            name: p["correlations"] for name, p in DISEASE_PROFILES.items()
        },
        "features_for_model": [
            "heart_rate", "hrv_rmssd", "spo2", "eda", "skin_temp",
            "respiration", "steps", "bp_systolic", "bp_diastolic",
            "sleep_stage", "sleep_quality", "hour_of_day"
        ],
        "target": "label_id",
        "secondary_target": "risk_score",
        "recommended_models": [
            "LSTM (sequence length = 12 samples = 1 hour)",
            "Transformer encoder (multi-head attention over signal correlations)",
            "1D-CNN + BiLSTM hybrid",
            "Temporal Fusion Transformer (TFT)",
        ],
        "recommended_window_size": 12,
        "recommended_stride": 6,
    }

    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  📋 Meta: {meta_path}")

    # ─── Print class summary ──────────────────────────────────────────────────
    print(f"\n  {'Label':<22} {'ID':>4} {'Subjects':>10} {'Samples':>12}")
    print(f"  {'─'*50}")
    for name, p in DISEASE_PROFILES.items():
        cnt = class_counts.get(name, 0)
        samples = cnt * hours * (60 // sample_rate_min)
        print(f"  {name:<22} {p['label_id']:>4} {cnt:>10} {samples:>12,}")

    print(f"\n  ✅ Dataset ready for training!\n")
    return csv_path, meta_path


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioRhythm Fusion Band — Training Data Generator")
    parser.add_argument("--subjects", type=int, default=500, help="Number of subjects (default: 500)")
    parser.add_argument("--hours", type=int, default=72, help="Hours per subject (default: 72)")
    parser.add_argument("--rate", type=int, default=5, help="Sample rate in minutes (default: 5)")
    parser.add_argument("--output", type=str, default="datasets/training", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 100 subjects, 24h")
    args = parser.parse_args()

    if args.quick:
        args.subjects = 100
        args.hours = 24

    generate_dataset(
        n_subjects=args.subjects,
        hours=args.hours,
        sample_rate_min=args.rate,
        output_dir=args.output,
    )
