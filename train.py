"""
train.py
--------
Training entry-point for the PAN 2026 Generated Plagiarism Detection system.

Usage
~~~~~

Train the lightweight feature-based classifier (fast, no GPU required)::

    python train.py \\
        --train data/raw/train.jsonl \\
        --dev   data/raw/dev.jsonl \\
        --mode  features \\
        --output results/feature_clf.joblib

Fine-tune DeBERTa on the training set::

    python train.py \\
        --train data/raw/train.jsonl \\
        --dev   data/raw/dev.jsonl \\
        --mode  finetune \\
        --model microsoft/deberta-v3-base \\
        --output results/deberta_model \\
        --epochs 3 \\
        --batch-size 8

Run the zero-shot perplexity baseline (no training needed)::

    python train.py \\
        --train data/raw/train.jsonl \\
        --mode  zeroshot \\
        --output results/zeroshot_threshold.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> List[dict]:
    from src.data_loader import load_jsonl
    log.info("Loading %s …", path)
    return load_jsonl(path)


def _texts_labels(records: List[dict]):
    texts = [r["text"] for r in records]
    labels = [r.get("label", 0) for r in records]
    source_texts = [r.get("source_text") for r in records]
    return texts, labels, source_texts


def _build_feature_matrix(texts: List[str], source_texts: List[str | None]) -> np.ndarray:
    """Extract all hand-crafted features and concatenate into a matrix."""
    from src.features.stylometric import StylometricFeatures
    from src.features.embeddings import EmbeddingFeatures

    log.info("Extracting stylometric features …")
    sty = StylometricFeatures()
    sty_feats = sty.extract_batch(texts)

    log.info("Extracting embedding features …")
    emb = EmbeddingFeatures()
    emb_feats = emb.extract_batch(texts, source_texts)

    rows = []
    for sf, ef in zip(sty_feats, emb_feats):
        vec = list(sf.values()) + [ef["self_sim"], ef["source_sim"]]
        # Replace NaN with 0
        vec = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in vec]
        rows.append(vec)

    feature_names = list(sty_feats[0].keys()) + ["self_sim", "source_sim"]
    return np.array(rows), feature_names


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def train_features(args: argparse.Namespace) -> None:
    from src.models.classifier import FeatureClassifier
    from src.evaluate import print_report

    train_records = _load(args.train)
    texts, labels, source_texts = _texts_labels(train_records)

    X_train, feat_names = _build_feature_matrix(texts, source_texts)

    clf = FeatureClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    log.info("Training FeatureClassifier …")
    clf.fit(X_train, labels, feature_names=feat_names)

    if args.dev:
        dev_records = _load(args.dev)
        dev_texts, dev_labels, dev_sources = _texts_labels(dev_records)
        X_dev, _ = _build_feature_matrix(dev_texts, dev_sources)
        preds = clf.predict(X_dev)
        y_pred = [p["label"] for p in preds]
        y_score = [p["score"] for p in preds]
        print_report(dev_labels, y_pred, y_score, title="Dev-set Evaluation (FeatureClassifier)")

    clf.save(args.output)
    log.info("Model saved to %s", args.output)


def train_finetune(args: argparse.Namespace) -> None:
    from src.models.classifier import FineTunedClassifier
    from src.evaluate import print_report

    train_records = _load(args.train)
    texts, labels, _ = _texts_labels(train_records)

    eval_texts, eval_labels = None, None
    if args.dev:
        dev_records = _load(args.dev)
        eval_texts, eval_labels, _ = _texts_labels(dev_records)

    clf = FineTunedClassifier(
        model_name=args.model,
        output_dir=args.output,
    )
    log.info("Fine-tuning %s …", args.model)
    clf.fit(
        texts, labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    if eval_texts is not None:
        preds = clf.predict(eval_texts)
        y_pred = [p["label"] for p in preds]
        y_score = [p["score"] for p in preds]
        print_report(eval_labels, y_pred, y_score, title="Dev-set Evaluation (FineTunedClassifier)")

    clf.save(args.output)
    log.info("Model saved to %s", args.output)


def train_zeroshot(args: argparse.Namespace) -> None:
    """Evaluate zero-shot detector and find optimal threshold."""
    from src.models.zero_shot import PerplexityThresholdDetector
    from src.evaluate import print_report, optimal_threshold

    train_records = _load(args.train)
    texts, labels, _ = _texts_labels(train_records)

    detector = PerplexityThresholdDetector()
    log.info("Running zero-shot perplexity detector …")
    preds = detector.predict(texts)
    y_score = [p["score"] for p in preds]
    y_pred = [p["label"] for p in preds]

    print_report(labels, y_pred, y_score, title="Zero-shot (PerplexityThreshold)")
    best_thresh, best_f1 = optimal_threshold(labels, y_score, metric="f1_macro")
    log.info("Optimal threshold: %.3f  (F1-macro=%.4f)", best_thresh, best_f1)

    output = {"threshold": best_thresh, "f1_macro": best_f1}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Threshold config saved to %s", args.output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a PAN 2026 Generated Plagiarism Detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train", required=True, help="Path to training JSONL file.")
    p.add_argument("--dev", default=None, help="Path to dev/validation JSONL file.")
    p.add_argument(
        "--mode",
        choices=["features", "finetune", "zeroshot"],
        default="features",
        help="Training mode.",
    )
    p.add_argument("--output", default="results/model", help="Output path for the saved model.")

    # FineTuned options
    p.add_argument("--model", default="microsoft/deberta-v3-base", help="HuggingFace model ID.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)

    # FeatureClassifier options
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "features":
        train_features(args)
    elif args.mode == "finetune":
        train_finetune(args)
    elif args.mode == "zeroshot":
        train_zeroshot(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
