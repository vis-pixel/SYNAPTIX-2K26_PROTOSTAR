#!/usr/bin/env python3
"""
BioRhythm X — Stress Scenario Simulator
Simulates a high-stress physiological scenario to test adaptive systems.

Usage: python scripts/simulate_stress.py [--token <JWT>] [--user-id <UUID>]
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.prediction_service import predict_biometrics
from app.anomaly_engine.detector import detect_anomalies
from app.diet_engine.macro_calculator import calculate_macros
from app.diet_engine.meal_planner import AdaptiveDietEngine


def print_band(title: str, char: str = "─", width: int = 65):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


def run_stress_scenario():
    print("🧠 BioRhythm X — Stress Scenario Simulation")
    print("Simulating a high-stress + overtraining + poor sleep state...")

    # ── Stressed vitals snapshot ──────────────────────────────────────────
    stressed_vitals = {
        "hr": 102.0,        # Elevated resting HR
        "hrv": 18.0,        # Critically low HRV (baseline ~55ms)
        "spo2": 96.0,       # Slightly low
        "temperature": 37.6,# Slightly elevated
        "gsr": 18.5,        # High skin conductance (stress)
        "respiration": 22.0,# Elevated breathing
        "accel_x": 0.05,
        "accel_y": 0.03,
        "accel_z": 9.8,
    }

    profile = {
        "resting_hr": 62.0,
        "baseline_hrv": 55.0,
        "max_hr": 195.0,      # 25 year old
        "weight_kg": 80.0,
        "height_cm": 180.0,
        "age": 25,
        "gender": "male",
        "activity_level": "very_active",
        "vo2_max": 52.0,
        "body_fat_pct": 15.0,
    }

    load = {
        "atl": 85.0,          # Very high acute load
        "ctl": 55.0,          # Chronic load (overreaching: ATL >> CTL)
        "fatigue_load": 85.0,
        "sleep_quality": 0.38,# Poor sleep
        "stress_level": 0.82, # High psychological stress
        "consecutive_training_days": 8,
        "diet_protein_g": 120.0,
    }

    print_band("🔴 VITALS SNAPSHOT")
    for k, v in stressed_vitals.items():
        print(f"  {k:20s}: {v}")

    # ── Anomaly Detection ─────────────────────────────────────────────────
    print_band("🤖 ANOMALY DETECTION (Ensemble)")
    anomalies = detect_anomalies([stressed_vitals])
    a = anomalies[0]
    print(f"  Is Anomaly:     {a['is_anomaly']}")
    print(f"  Anomaly Score:  {a['anomaly_score']:.3f}")
    print(f"  Severity:       {a['severity'].upper()}")
    print(f"  Anomaly Type:   {a['anomaly_type']}")
    print(f"  Violations:     {a.get('violations', [])}")

    # ── Biometric Predictions ─────────────────────────────────────────────
    print_band("📊 BIOMETRIC PREDICTIONS (8-Dimensional)")
    preds = predict_biometrics(stressed_vitals, profile, load)
    display = {
        "Illness Onset Probability": f"{preds['illness_onset_probability']:.0%}",
        "Overtraining Risk":         f"{preds['overtraining_risk']:.0%}",
        "Recovery Time":             f"{preds['recovery_time_hours']:.0f} hours",
        "Dehydration Risk":          f"{preds['dehydration_risk']:.0%}",
        "Cardiovascular Strain":     f"{preds['cardiovascular_strain']:.0%}",
        "Metabolic Flexibility":     f"{preds['metabolic_flexibility_score']:.0%}",
        "Hormonal Stress Proxy":     f"{preds['hormonal_stress_proxy']:.0%}",
        "Overall Risk Score":        f"{preds['overall_risk']:.0%}",
    }
    for k, v in display.items():
        print(f"  {k:30s}: {v}")

    print(f"\n  🧪 Nutrient Deficiency Risk:")
    for nut, score in preds["nutrient_deficiency_risk"].items():
        bar = "●" * round(score * 10) + "○" * (10 - round(score * 10))
        print(f"    {nut:15s}: [{bar}] {score:.0%}")

    # ── Adaptive Diet ─────────────────────────────────────────────────────
    print_band("🥗 ADAPTIVE DIET RESPONSE")
    macros = calculate_macros(
        weight_kg=80, height_cm=180, age=25, gender="male",
        activity_level="very_active", goal="muscle_gain"
    )
    adaptive = AdaptiveDietEngine()
    adapted = adaptive.adapt(
        macros=macros,
        hrv=stressed_vitals["hrv"],
        baseline_hrv=profile["baseline_hrv"],
        stress_level=load["stress_level"],
        sleep_quality=load["sleep_quality"],
        fatigue_load=load["fatigue_load"],
        illness_probability=preds["illness_onset_probability"],
    )
    print(f"  Base Calories:  {macros.calorie_target:.0f} kcal")
    print(f"  Adapted:        {adapted['calorie_target']:.0f} kcal (+{adapted['calorie_target'] - macros.calorie_target:.0f})")
    print(f"  Adapted Carbs:  {adapted['carbs_g']:.0f}g (was {macros.carbs_g:.0f}g)")
    print(f"  Extra Water:    {adapted['water_ml']:.0f}ml")
    print(f"\n  📋 Adjustments:")
    for reason in adapted["adjustment_reasons"]:
        print(f"    → {reason}")
    print(f"\n  🍊 Recommended Foods:")
    for food in adapted["food_recommendations"]:
        print(f"    → {food}")

    print(f"\n{'='*65}")
    print("  ✅ Stress scenario simulation complete!")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_stress_scenario()
