#!/usr/bin/env python3
"""
BioRhythm X — Train All ML Models
Usage: python scripts/train_models.py [--dataset mit_bih] [--synthetic]

Models trained:
- IsolationForest (anomaly detection)
- LSTM (sequence prediction)
- Autoencoder (reconstruction anomaly detection)
"""
import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ml.models import BioIsolationForest, BioLSTM, BioAutoencoderWrapper
from app.synthetic_generator.generator import generate_training_dataset
from app.dataset_loader.parser import parse_auto, find_parseable_files
from app.dataset_loader.normalizer import normalize_batch
from app.dataset_loader.registry import get_dataset_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def load_dataset(dataset_name: str, max_samples: int = 50000) -> list[dict]:
    """Load and normalize a real dataset from disk."""
    cfg = get_dataset_config(dataset_name)
    if not cfg:
        logger.error(f"Unknown dataset: {dataset_name}")
        return []
    local_dir = settings.get_dataset_path(cfg.local_dir)
    files = find_parseable_files(str(local_dir), dataset_name)
    if not files:
        logger.warning(f"No parseable files found in {local_dir}")
        return []
    data = []
    for f in files[:5]:   # Limit to first 5 files for speed
        rows = parse_auto(f, dataset_name, max_samples=max_samples // 5)
        data.extend(rows)
        if len(data) >= max_samples:
            break
    return normalize_batch(data, dataset_name=dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Train BioRhythm X ML models")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name (e.g. mit_bih)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--model", type=str, default="all", help="Model: all|isolation_forest|lstm|autoencoder")
    args = parser.parse_args()

    # ── Load Training Data ────────────────────────────────────────────────
    if args.synthetic or not args.dataset:
        logger.info(f"Generating {args.samples} synthetic training samples...")
        data = generate_training_dataset(n_per_activity=args.samples // 5)
        logger.info(f"Generated {len(data)} synthetic samples")
    else:
        logger.info(f"Loading dataset: {args.dataset}")
        data = load_dataset(args.dataset, max_samples=args.samples)
        if not data:
            logger.warning("Dataset empty or not found — falling back to synthetic data")
            data = generate_training_dataset(n_per_activity=args.samples // 5)

    if len(data) < 100:
        logger.error("Not enough training data (need at least 100 samples)")
        sys.exit(1)

    logger.info(f"Training on {len(data)} samples")

    # ── Split 80/20 ────────────────────────────────────────────────────────
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    # ── Create models dir ─────────────────────────────────────────────────
    Path(settings.ML_MODELS_DIR).mkdir(parents=True, exist_ok=True)

    train_all = args.model == "all"

    # ── Train IsolationForest ─────────────────────────────────────────────
    if train_all or args.model == "isolation_forest":
        logger.info("Training IsolationForest...")
        iso = BioIsolationForest()
        iso.train(train_data)
        iso.save()
        # Quick eval
        preds = iso.predict(val_data[:100])
        n_anomalies = sum(1 for p in preds if p["is_anomaly"])
        logger.info(f"IsolationForest: {n_anomalies}/{len(preds)} anomalies in validation")

    # ── Train Autoencoder ─────────────────────────────────────────────────
    if train_all or args.model == "autoencoder":
        logger.info("Training Autoencoder...")
        ae = BioAutoencoderWrapper()
        ae.train(train_data, epochs=settings.AUTOENCODER_EPOCHS)
        ae.save()
        preds = ae.predict(val_data[:100])
        n_anomalies = sum(1 for p in preds if p["is_anomaly"])
        logger.info(f"Autoencoder threshold={ae.threshold:.6f} | {n_anomalies}/{len(preds)} anomalies")

    # ── Train LSTM ────────────────────────────────────────────────────────
    if train_all or args.model == "lstm":
        if len(train_data) >= 100:
            logger.info("Training LSTM...")
            lstm = BioLSTM(seq_len=30)
            lstm.train(train_data, epochs=settings.LSTM_EPOCHS)
            lstm.save()
            logger.info("LSTM training complete")
        else:
            logger.warning("Not enough data for LSTM (need 100+ samples)")

    logger.info("✅ All models trained and saved to " + settings.ML_MODELS_DIR)


if __name__ == "__main__":
    main()
