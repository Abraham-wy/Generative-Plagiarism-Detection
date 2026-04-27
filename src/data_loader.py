"""
data_loader.py
--------------
Utilities for loading and saving PAN 2026 Generated-Plagiarism-Detection
datasets.

Expected input formats
~~~~~~~~~~~~~~~~~~~~~~
JSONL (one JSON object per line)::

    {"id": "doc1", "text": "The quick brown fox ...", "label": 1}
    {"id": "doc2", "text": "Lorem ipsum ...",          "label": 0}

For paired tasks a ``source_text`` field may also be present::

    {"id": "doc3", "text": "...", "source_text": "...", "label": 1}

``label`` is optional in test sets (prediction mode).

XML (PAN corpus format)
~~~~~~~~~~~~~~~~~~~~~~~
Each document is a separate file ``<id>.txt`` inside a directory, with a
companion ``truth.jsonl`` (or ``truth.xml``) for labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_record(raw: dict) -> dict:
    """Normalise a raw JSON record into the canonical schema."""
    record: dict = {
        "id": str(raw.get("id", "")),
        "text": str(raw.get("text", "")),
        "source_text": str(raw.get("source_text", "")) if raw.get("source_text") else None,
    }
    if "label" in raw:
        record["label"] = int(raw["label"])
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[dict]:
    """Load a JSONL file and return a list of normalised record dicts."""
    records: List[dict] = []
    with jsonlines.open(str(path)) as reader:
        for raw in reader:
            records.append(_parse_record(raw))
    return records


def load_directory(
    text_dir: str | Path,
    truth_path: Optional[str | Path] = None,
) -> List[dict]:
    """Load a directory of plain-text files with an optional truth file.

    Parameters
    ----------
    text_dir:
        Directory containing ``<id>.txt`` files.
    truth_path:
        Optional path to a JSONL truth file mapping ``id`` → ``label``.

    Returns
    -------
    List of record dicts (same schema as :func:`load_jsonl`).
    """
    text_dir = Path(text_dir)
    labels: Dict[str, int] = {}
    if truth_path is not None:
        with jsonlines.open(str(truth_path)) as reader:
            for row in reader:
                labels[str(row["id"])] = int(row["label"])

    records: List[dict] = []
    for txt_file in sorted(text_dir.glob("*.txt")):
        doc_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        record: dict = {"id": doc_id, "text": text, "source_text": None}
        if doc_id in labels:
            record["label"] = labels[doc_id]
        records.append(record)
    return records


def to_dataframe(records: List[dict]) -> pd.DataFrame:
    """Convert a list of record dicts to a :class:`pandas.DataFrame`."""
    return pd.DataFrame(records)


def save_predictions(
    predictions: List[dict],
    output_path: str | Path,
) -> None:
    """Save predictions to a JSONL file.

    Each entry in *predictions* should have at least ``id`` and ``score``
    (a float in [0, 1] representing the probability of AI generation).
    A ``label`` key (0 or 1) is added automatically via hard threshold 0.5.

    Parameters
    ----------
    predictions:
        List of dicts with keys ``id`` and ``score``.
    output_path:
        Destination file path (will be created / overwritten).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for pred in predictions:
            record = {
                "id": pred["id"],
                "score": float(pred["score"]),
                "label": int(pred["score"] >= 0.5),
            }
            writer.write(record)


def load_predictions(path: str | Path) -> List[dict]:
    """Load a previously saved predictions JSONL file."""
    with jsonlines.open(str(path)) as reader:
        return list(reader)
