"""
BioRhythm Fusion Band — Smartwatch CSV Bridge
═══════════════════════════════════════════════
Reads health data from Da Fit / Fire-Boltt CSV export
and sends it to the backend API for LSTM risk analysis.

Flow:
  Da Fit App → Export CSV → This script → POST /live-data → Risk Score

Usage:
    python scripts/watch_bridge.py --csv data/dafit_export.csv
    python scripts/watch_bridge.py --manual
    python scripts/watch_bridge.py --csv data/dafit_export.csv --api http://localhost:5000/live-data
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


API_URL = "http://localhost:5000/live-data"


def send_to_api(data, api_url=API_URL):
    """Send health data to the risk prediction API."""
    payload = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            return result
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None


def process_csv(csv_path, api_url, delay=1.0):
    """Read CSV file and send each row to the API."""
    print(f"\n{'='*60}")
    print(f"  🔗 Smartwatch CSV Bridge")
    print(f"{'='*60}")
    print(f"  CSV: {csv_path}")
    print(f"  API: {api_url}")
    print(f"{'='*60}\n")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Found {len(rows)} readings in CSV\n")

    # Column mapping (Da Fit / Fire-Boltt variante)
    col_map = {
        "heart_rate":   ["heart_rate", "hr", "HeartRate", "Heart Rate", "pulse"],
        "spo2":         ["spo2", "SpO2", "blood_oxygen", "Blood Oxygen", "oxygen"],
        "temperature":  ["temperature", "temp", "Temperature", "skin_temp", "Skin Temp"],
        "sleep_hours":  ["sleep_hours", "sleep", "Sleep", "SleepHours", "sleep_duration"],
        "hrv_rmssd":    ["hrv_rmssd", "hrv", "HRV", "rmssd"],
        "respiration":  ["respiration", "resp", "Respiration", "breathing_rate"],
        "bp_systolic":  ["bp_systolic", "bp_sys", "systolic", "Systolic"],
        "bp_diastolic": ["bp_diastolic", "bp_dia", "diastolic", "Diastolic"],
    }

    headers = list(rows[0].keys()) if rows else []
    field_map = {}
    for target, candidates in col_map.items():
        for c in candidates:
            if c in headers:
                field_map[target] = c
                break

    print(f"  Column mapping: {json.dumps(field_map, indent=4)}\n")
    print(f"  {'#':>4}  {'HR':>6}  {'SpO2':>6}  {'Temp':>6}  {'Risk':>6}  {'Alert':<12}")
    print(f"  {'─'*50}")

    for i, row in enumerate(rows):
        data = {}
        for target, csv_col in field_map.items():
            try:
                data[target] = float(row.get(csv_col, 0))
            except (ValueError, TypeError):
                data[target] = 0
        data["skin_temp"] = data.get("temperature", 36.5)

        result = send_to_api(data, api_url)
        if result:
            risk = result["risk_score"]
            level = result["alert_level"]
            marker = "🟢" if level == "Normal" else "🟡" if level == "Warning" else "🔴"
            print(f"  {i+1:>4}  {data.get('heart_rate',0):>6.0f}  {data.get('spo2',0):>6.1f}  {data.get('temperature',0):>6.1f}  {risk:>5}%  {marker} {level}")

            if level == "Critical":
                print(f"         ⚠️  CRITICAL ALERT — Risk {risk}% — Check patient immediately!")
        else:
            print(f"  {i+1:>4}  — API unreachable, skipping")

        time.sleep(delay)

    print(f"\n  ✅ Processed {len(rows)} readings\n")


def manual_entry(api_url):
    """Interactive manual data entry mode."""
    print(f"\n{'='*60}")
    print(f"  🔗 Manual Health Data Entry")
    print(f"{'='*60}")
    print(f"  API: {api_url}")
    print(f"  Type 'quit' to exit\n")

    while True:
        try:
            print(f"\n  ─── New Reading ───")
            hr = input("  ❤️  Heart Rate (BPM):     ").strip()
            if hr.lower() == 'quit':
                break
            spo2 = input("  🫀 SpO₂ (%):             ").strip()
            temp = input("  🌡️  Temperature (°C):     ").strip()
            sleep = input("  🌙 Sleep Hours:          ").strip()

            data = {
                "heart_rate":  float(hr or 72),
                "spo2":        float(spo2 or 98),
                "temperature": float(temp or 36.5),
                "skin_temp":   float(temp or 36.5),
                "sleep_hours": float(sleep or 7),
                "hrv_rmssd":   55,
                "respiration": 15,
                "bp_systolic": 120,
                "bp_diastolic": 78,
            }

            result = send_to_api(data, api_url)
            if result:
                risk = result["risk_score"]
                level = result["alert_level"]
                color = "🟢" if level == "Normal" else "🟡" if level == "Warning" else "🔴"
                print(f"\n  {color} Risk Score: {risk}% — {level}")
                if level == "Critical":
                    print(f"  🚨 CRITICAL ALERT! Seek medical attention!")
            else:
                print(f"\n  ❌ Could not reach API. Is the server running?")

        except (KeyboardInterrupt, EOFError):
            break

    print(f"\n  Goodbye! Stay healthy. 💚\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smartwatch CSV Bridge")
    parser.add_argument("--csv", type=str, help="Path to Da Fit / Fire-Boltt CSV export")
    parser.add_argument("--manual", action="store_true", help="Manual data entry mode")
    parser.add_argument("--api", type=str, default=API_URL, help="API endpoint URL")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between rows (seconds)")
    args = parser.parse_args()

    if args.csv:
        process_csv(args.csv, args.api, args.delay)
    elif args.manual:
        manual_entry(args.api)
    else:
        print("Usage:")
        print("  python scripts/watch_bridge.py --csv path/to/export.csv")
        print("  python scripts/watch_bridge.py --manual")
        print("  python scripts/watch_bridge.py --csv data.csv --api http://localhost:5000/live-data")
