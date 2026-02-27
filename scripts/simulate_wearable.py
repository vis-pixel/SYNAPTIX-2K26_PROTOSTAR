#!/usr/bin/env python3
"""
BioRhythm X — Wearable Simulator
Sends synthetic wearable data to the WebSocket endpoint.

Usage:
  python scripts/simulate_wearable.py --user-id <UUID> --token <JWT>
  python scripts/simulate_wearable.py --user-id <UUID> --token <JWT> --activity Run --samples 500
"""
import asyncio
import argparse
import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets
from app.synthetic_generator.generator import generate_vitals_snapshot


async def simulate(user_id: str, token: str, host: str, activity: str, n_samples: int):
    uri = f"ws://{host}/ws/vitals/{user_id}"
    print(f"🔌 Connecting to {uri}...")

    async with websockets.connect(
        uri,
        extra_headers={"Authorization": f"Bearer {token}"},
    ) as ws:
        print(f"✅ Connected. Sending {n_samples} {activity} samples...")
        anomaly_count = 0
        for i in range(n_samples):
            stress = random.uniform(0.1, 0.5)
            snap = generate_vitals_snapshot(activity=activity, stress_level=stress)
            await ws.send(json.dumps(snap))
            resp = json.loads(await ws.recv())
            if resp.get("is_anomaly"):
                anomaly_count += 1
                print(f"  ⚠️  Sample {i+1}: ANOMALY [{resp['severity']}] {resp.get('anomaly_type','')} score={resp.get('anomaly_score'):.3f}")
            elif (i + 1) % 50 == 0:
                print(f"  ✔ Sample {i+1}/{n_samples} OK")
            await asyncio.sleep(0.02)  # ~50 Hz

        print(f"\n📊 Done! {n_samples} samples sent. {anomaly_count} anomalies detected ({anomaly_count/n_samples*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="BioRhythm X Wearable Simulator")
    parser.add_argument("--user-id", required=True, help="User UUID")
    parser.add_argument("--token", required=True, help="JWT access token")
    parser.add_argument("--host", default="localhost:8000", help="API host:port")
    parser.add_argument("--activity", default="Walk", choices=["Rest", "Walk", "Run", "Sprint", "Gym"])
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to send")
    args = parser.parse_args()
    asyncio.run(simulate(args.user_id, args.token, args.host, args.activity, args.samples))


if __name__ == "__main__":
    main()
