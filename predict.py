"""
predict.py
----------
Inference entry-point for the PAN 2026 Generated Plagiarism Detection system.

Usage
~~~~~

Predict with a feature-based classifier::

    python predict.py \\
        --input  data/raw/test.jsonl \\
        --model  results/feature_clf.joblib \\
        --mode   features \\
        --output results/predictions.jsonl

Predict with a fine-tuned transformer::

    python predict.py \\
        --input  data/raw/test.jsonl \\
        --model  results/deberta_model \\
        --mode   finetune \\
        --output results/predictions.jsonl

Predict with the zero-shot perplexity baseline::

    python predict.py \\
        --input  data/raw/test.jsonl \\
        --mode   zeroshot \\
        --threshold 0.4 \\
        --output results/predictions.jsonl

Predict with the RoBERTa zero-shot detector::

    python predict.py \\
        --input  data/raw/test.jsonl \\
        --mode   roberta \\
        --output results/predictions.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
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


def _texts_sources(records: List[dict]):
    texts = [r["text"] for r in records]
    source_texts = [r.get("source_text") for r in records]
    return texts, source_texts


def _build_feature_matrix(texts: List[str], source_texts: List[str | None]) -> np.ndarray:
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
        vec = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in vec]
        rows.append(vec)
    return np.array(rows)


def _attach_ids(records: List[dict], preds: List[dict]) -> List[dict]:
    return [{"id": r["id"], **p} for r, p in zip(records, preds)]


# ---------------------------------------------------------------------------
# Prediction modes
# ---------------------------------------------------------------------------

def predict_features(args: argparse.Namespace) -> None:
    from src.models.classifier import FeatureClassifier
    from src.data_loader import save_predictions

    records = _load(args.input)
    texts, source_texts = _texts_sources(records)
    X = _build_feature_matrix(texts, source_texts)

    clf = FeatureClassifier()
    clf.load(args.model)
    preds = clf.predict(X)
    preds_with_ids = _attach_ids(records, preds)

    save_predictions(preds_with_ids, args.output)
    log.info("Predictions saved to %s", args.output)


def predict_finetune(args: argparse.Namespace) -> None:
    from src.models.classifier import FineTunedClassifier
    from src.data_loader import save_predictions

    records = _load(args.input)
    texts, _ = _texts_sources(records)

    clf = FineTunedClassifier()
    clf.load(args.model)
    preds = clf.predict(texts)
    preds_with_ids = _attach_ids(records, preds)

    save_predictions(preds_with_ids, args.output)
    log.info("Predictions saved to %s", args.output)


def predict_zeroshot(args: argparse.Namespace) -> None:
    from src.models.zero_shot import PerplexityThresholdDetector
    from src.data_loader import save_predictions

    records = _load(args.input)
    texts, _ = _texts_sources(records)

    detector = PerplexityThresholdDetector(threshold=args.threshold)
    log.info("Running zero-shot perplexity detector (threshold=%.3f) …", args.threshold)
    preds = detector.predict(texts)
    preds_with_ids = _attach_ids(records, preds)

    save_predictions(preds_with_ids, args.output)
    log.info("Predictions saved to %s", args.output)


def predict_roberta(args: argparse.Namespace) -> None:
    from src.models.zero_shot import RobertaDetector
    from src.data_loader import save_predictions

    records = _load(args.input)
    texts, _ = _texts_sources(records)

    detector = RobertaDetector(model_name=args.model or "openai-community/roberta-base-openai-detector")
    log.info("Running RoBERTa zero-shot detector …")
    preds = detector.predict(texts)
    preds_with_ids = _attach_ids(records, preds)

    save_predictions(preds_with_ids, args.output)
    log.info("Predictions saved to %s", args.output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference for PAN 2026 Generated Plagiarism Detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input JSONL file.")
    p.add_argument("--output", required=True, help="Path to output predictions JSONL file.")
    p.add_argument(
        "--mode",
        choices=["features", "finetune", "zeroshot", "roberta"],
        default="features",
        help="Inference mode.",
    )
    p.add_argument("--model", default=None, help="Path to saved model or HuggingFace ID.")
    p.add_argument("--threshold", type=float, default=50.0,
                   help="Perplexity threshold (zeroshot mode only).")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "features":
        if not args.model:
            parser.error("--model is required for features mode.")
        predict_features(args)
    elif args.mode == "finetune":
        if not args.model:
            parser.error("--model is required for finetune mode.")
        predict_finetune(args)
    elif args.mode == "zeroshot":
        predict_zeroshot(args)
    elif args.mode == "roberta":
        predict_roberta(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
