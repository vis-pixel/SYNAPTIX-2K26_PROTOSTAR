#!/usr/bin/env python3
"""
BioRhythm X — Per-Field Dataset Training Pipeline
================================================
Trains ONE ML model per health field using its dedicated dataset.

Field → Dataset → Model mapping:
  ecg              ← MIT-BIH    → autoencoder
  hrv              ← Fantasia   → isolation_forest
  hrv_balance_index← Fantasia   → isolation_forest
  spo2             ← BIDMC      → isolation_forest
  blood_flow_waveform← BIDMC    → autoencoder
  respiration      ← BIDMC      → isolation_forest
  eda_level        ← WESAD      → isolation_forest
  sweat_chemical   ← WESAD      → isolation_forest
  skin_temperature ← WESAD      → isolation_forest
  stress_level     ← WESAD      → autoencoder
  sleep_stage      ← Sleep-EDF  → lstm
  sleep_frag_index ← Sleep-EDF  → isolation_forest
  circadian_score  ← Sleep-EDF  → isolation_forest
  accel_x          ← MHEALTH    → isolation_forest
  activity_label   ← MHEALTH    → isolation_forest

Usage:
  # Train ALL fields (uses synthetic fallback if dataset missing)
  python scripts/train_field_models.py

  # Train ONE field
  python scripts/train_field_models.py --field ecg

  # Force use synthetic data only
  python scripts/train_field_models.py --synthetic
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_fields")

from app.config import settings
from app.dataset_loader.field_mapping import FIELD_DATASET_MAP, get_map_for_field
from app.dataset_loader.parser import parse_auto, find_parseable_files
from app.dataset_loader.normalizer import normalize_batch
from app.dataset_loader.registry import get_dataset_config
from app.ml.field_models import get_field_model
from app.synthetic_generator.generator import generate_session


MODELS_DIR = settings.ML_MODELS_DIR
MAX_SAMPLES_PER_FIELD = 8000
AE_EPOCHS = 40


# ─── Load real dataset for a field ────────────────────────────────────────────
def load_field_data(mapping, max_samples: int = MAX_SAMPLES_PER_FIELD) -> list[dict]:
    cfg = get_dataset_config(mapping.dataset_name)
    if not cfg:
        return []

    local_dir = Path(settings.DATASET_BASE_DIR) / cfg.local_dir
    if not local_dir.exists():
        logger.warning(f"[{mapping.field_name}] Dataset dir not found: {local_dir}")
        return []

    files = find_parseable_files(str(local_dir), mapping.dataset_name)
    if not files:
        logger.warning(f"[{mapping.field_name}] No parseable files in {local_dir}")
        return []

    raw = []
    per_file = max(500, max_samples // max(1, len(files[:10])))
    for f in files[:10]:
        rows = parse_auto(f, mapping.dataset_name, max_samples=per_file)
        raw.extend(rows)
        if len(raw) >= max_samples:
            break

    normalized = normalize_batch(raw, dataset_name=mapping.dataset_name)
    logger.info(f"[{mapping.field_name}] Loaded {len(normalized)} samples from {mapping.dataset_name}")
    return normalized


# ─── Synthetic fallback data ───────────────────────────────────────────────────
def get_synthetic_fallback(field_name: str, n: int = 3000) -> list[dict]:
    """Generate synthetic data as fallback when real dataset not available."""
    data = []
    for activity in ["Rest", "Walk", "Run", "Gym"]:
        session = generate_session(n_samples=n // 4, activity=activity, noise_anomalies=True)
        data.extend(session)
    return data


# ─── Train ONE field ───────────────────────────────────────────────────────────
def train_one_field(mapping, force_synthetic: bool = False) -> dict:
    start = time.time()
    field = mapping.field_name
    logger.info(f"\n{'='*55}")
    logger.info(f"  FIELD: {field}  |  Dataset: {mapping.dataset_name}  |  Model: {mapping.model_type}")
    logger.info(f"{'='*55}")

    # Load data
    if force_synthetic:
        data = get_synthetic_fallback(field)
        source = "synthetic"
    else:
        data = load_field_data(mapping)
        source = mapping.dataset_name
        if len(data) < 50:
            logger.warning(f"[{field}] Real data insufficient — falling back to synthetic")
            data = get_synthetic_fallback(field)
            source = "synthetic_fallback"

    if len(data) < 10:
        return {"field": field, "status": "skipped", "reason": "no_data"}

    # Build and train model
    model = get_field_model(field, mapping.model_type, MODELS_DIR)
    try:
        if mapping.model_type == "autoencoder":
            model.train(data, epochs=AE_EPOCHS)
        else:
            model.train(data)
        model.save()

        # Quick eval
        sample = data[:min(200, len(data))]
        preds = model.predict(sample)
        n_anomalies = sum(1 for p in preds if p["is_anomaly"])
        elapsed = round(time.time() - start, 2)

        logger.info(f"[{field}] DONE | source={source} | samples={len(data)} | "
                    f"anomalies={n_anomalies}/{len(sample)} | time={elapsed}s")
        return {
            "field": field,
            "status": "trained",
            "dataset": mapping.dataset_name,
            "source": source,
            "samples": len(data),
            "model_type": mapping.model_type,
            "validation_anomalies": f"{n_anomalies}/{len(sample)}",
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        logger.error(f"[{field}] Training failed: {e}")
        return {"field": field, "status": "error", "error": str(e)}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BioRhythm X — Per-Field Model Training")
    parser.add_argument("--field", type=str, default="", help="Train only this field")
    parser.add_argument("--synthetic", action="store_true", help="Force use synthetic data")
    parser.add_argument("--list", action="store_true", help="List all field-dataset mappings")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Field':<25} {'Dataset':<12} {'Model':<20} {'Alert'}")
        print("-" * 85)
        for m in FIELD_DATASET_MAP:
            print(f"{m.field_name:<25} {m.dataset_name:<12} {m.model_type:<20} {m.alert_type}")
        return

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Select mappings
    if args.field:
        mapping = get_map_for_field(args.field)
        if not mapping:
            logger.error(f"Unknown field: '{args.field}'. Use --list to see all fields.")
            sys.exit(1)
        mappings = [mapping]
    else:
        mappings = FIELD_DATASET_MAP

    logger.info(f"\nTraining {len(mappings)} per-field models...")
    results = []
    for m in mappings:
        result = train_one_field(m, force_synthetic=args.synthetic)
        results.append(result)

    # Summary report
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE — {len(mappings)} fields")
    print(f"{'='*55}")
    print(f"{'Field':<25} {'Status':<12} {'Source':<20} {'Samples'}")
    print("-" * 75)
    for r in results:
        print(f"{r['field']:<25} {r.get('status','?'):<12} "
              f"{r.get('source',''):<20} {r.get('samples','')}")

    trained = sum(1 for r in results if r["status"] == "trained")
    print(f"\nTrained: {trained}/{len(mappings)} models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
