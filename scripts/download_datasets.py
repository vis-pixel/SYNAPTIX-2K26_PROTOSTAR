#!/usr/bin/env python3
"""
BioRhythm X — Standalone Dataset Downloader
Downloads all 6 research datasets directly without needing the API server.
No Docker, no JWT token needed — just run this script.

Usage:
    python scripts/download_datasets.py              # Download all
    python scripts/download_datasets.py --name mhealth   # Download one
    python scripts/download_datasets.py --list       # Show all datasets
"""
import argparse
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

BASE_DIR = Path(__file__).parent.parent / "datasets"

# ─── Dataset Catalog ──────────────────────────────────────────────────────────
DATASETS = {
    "mhealth": {
        "name": "MHEALTH — Activity Recognition",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
        "size": "~10 MB",
        "local_dir": "mhealth",
        "archive": "mhealth.zip",
        "fields": ["accel_x/y/z", "activity_label"],
        "license": "CC BY 4.0",
    },
    "mit_bih": {
        "name": "MIT-BIH Arrhythmia Database",
        "url": "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip",
        "size": "~100 MB",
        "local_dir": "mit_bih",
        "archive": "mitdb.zip",
        "fields": ["ecg"],
        "license": "ODC-BY 1.0",
    },
    "fantasia": {
        "name": "Fantasia Database (HRV)",
        "url": "https://physionet.org/static/published-projects/fantasia/fantasia-database-1.0.0.zip",
        "size": "~50 MB",
        "local_dir": "fantasia",
        "archive": "fantasia.zip",
        "fields": ["hrv", "hrv_balance_index"],
        "license": "ODC-BY 1.0",
    },
    "bidmc": {
        "name": "BIDMC PPG and Respiration Dataset",
        "url": "https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip",
        "size": "~200 MB",
        "local_dir": "bidmc",
        "archive": "bidmc.zip",
        "fields": ["spo2", "blood_flow_waveform", "respiration"],
        "license": "ODC-BY 1.0",
    },
    "sleep_edf": {
        "name": "Sleep-EDF Expanded Dataset",
        "url": "https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip",
        "size": "~1 GB",
        "local_dir": "sleep_edf",
        "archive": "sleep_edf.zip",
        "fields": ["sleep_stage", "sleep_fragmentation_index", "circadian_balance_score"],
        "license": "ODC-BY 1.0",
    },
    "wesad": {
        "name": "WESAD — Wearable Stress Detection",
        "url": "https://storage.googleapis.com/wesad-dataset/WESAD.zip",
        "size": "~1.7 GB",
        "local_dir": "wesad",
        "archive": "wesad.zip",
        "fields": ["eda_level", "sweat_chemical_level", "skin_temperature", "stress_level"],
        "license": "CC BY 4.0",
        "note": "Large file — 1.7 GB, may take 10-20 min on slow connection",
    },
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        filled = int(pct // 5)
        bar = "█" * filled + "░" * (20 - filled)
        mb_done = downloaded / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"\r  [{bar}] {pct:5.1f}% — {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
    else:
        mb = block_num * block_size / 1_000_000
        print(f"\r  Downloaded {mb:.1f} MB...", end="", flush=True)


def download_one(key: str) -> bool:
    cfg = DATASETS[key]
    local_dir = BASE_DIR / cfg["local_dir"]
    archive_path = BASE_DIR / cfg["archive"]

    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"  Fields: {', '.join(cfg['fields'])}")
    print(f"  Size:   {cfg['size']}")
    print(f"  License: {cfg['license']}")
    if "note" in cfg:
        print(f"  NOTE: {cfg['note']}")
    print(f"{'='*60}")

    # Already extracted?
    if local_dir.exists() and any(local_dir.iterdir()):
        files = list(local_dir.rglob("*"))
        print(f"  Already downloaded ({len(files)} files in {local_dir})")
        print(f"  Skip. Delete {local_dir} to re-download.")
        return True

    # Download archive
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from: {cfg['url']}")
    start = time.time()

    try:
        urlretrieve(cfg["url"], str(archive_path), reporthook=_progress)
        print()  # newline after progress bar
        elapsed = time.time() - start
        size_mb = archive_path.stat().st_size / 1_000_000
        print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.0f}s")
    except (URLError, HTTPError) as e:
        print(f"\n  ERROR downloading {key}: {e}")
        print(f"  Try manually: download from {cfg['url']}")
        print(f"  and extract to: {local_dir}")
        if archive_path.exists():
            archive_path.unlink()
        return False

    # Extract
    print(f"  Extracting to {local_dir}...")
    try:
        local_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(str(archive_path), "r") as z:
            total = len(z.namelist())
            for i, member in enumerate(z.namelist(), 1):
                z.extract(member, str(local_dir))
                if i % max(1, total // 20) == 0:
                    print(f"\r  Extracting {i}/{total} files...", end="", flush=True)
        print(f"\r  Extracted {total} files OK" + " " * 20)
        archive_path.unlink()  # Remove zip to save space
        print(f"  Ready → {local_dir}")
        return True
    except Exception as e:
        print(f"\n  ERROR extracting: {e}")
        return False


def show_list():
    print(f"\n{'Dataset':<12} {'Size':<10} {'Fields':<45} {'License'}")
    print("-" * 90)
    for key, cfg in DATASETS.items():
        status = ""
        d = BASE_DIR / cfg["local_dir"]
        if d.exists() and any(d.iterdir()):
            status = "[DOWNLOADED]"
        print(f"{key:<12} {cfg['size']:<10} {', '.join(cfg['fields']):<45} {status}")


def main():
    parser = argparse.ArgumentParser(description="BioRhythm X Dataset Downloader")
    parser.add_argument("--name", type=str, default="", help="Dataset key to download (e.g. mhealth)")
    parser.add_argument("--list", action="store_true", help="List all datasets and their status")
    parser.add_argument("--small-only", action="store_true", help="Only download small datasets (<300MB)")
    args = parser.parse_args()

    if args.list:
        show_list()
        return

    if args.name:
        if args.name not in DATASETS:
            print(f"Unknown dataset: '{args.name}'. Use --list to see options.")
            sys.exit(1)
        targets = [args.name]
    elif args.small_only:
        targets = ["mhealth", "fantasia", "mit_bih"]  # <200 MB each
        print("Downloading small datasets only (mhealth + fantasia + mit_bih)...")
    else:
        # Download in size order (smallest first)
        targets = ["mhealth", "fantasia", "mit_bih", "bidmc", "sleep_edf", "wesad"]
        print("Downloading ALL 6 datasets...")
        print("Total size: ~3 GB — this may take 15-45 minutes depending on connection speed.")

    print(f"\nSaving to: {BASE_DIR.resolve()}\n")
    results = {}
    for key in targets:
        results[key] = download_one(key)

    # Summary
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for key, ok in results.items():
        icon = "OK " if ok else "FAIL"
        print(f"  {icon}  {key:<12}  {DATASETS[key]['name']}")

    ok_count = sum(1 for v in results.values() if v)
    print(f"\n{ok_count}/{len(results)} datasets ready.")

    if ok_count > 0:
        print(f"\nNext step — Train per-field models:")
        print(f"  python scripts/train_field_models.py")
        print(f"\nOr train one field:")
        print(f"  python scripts/train_field_models.py --field ecg")


if __name__ == "__main__":
    main()
