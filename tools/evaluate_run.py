#!/usr/bin/env python3
"""
Evaluate a TREC run with nDCG@10, Recall@10, Recall@100, and RR.

This is useful when qrels are available. The official hidden test qrels are not
available before the deadline, so use this for spot-check or past datasets only.
Expected qrels format: qid 0 doc_id relevance
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import defaultdict
from pathlib import Path


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: defaultdict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            parts = raw_line.split()
            if len(parts) < 4:
                raise ValueError(f"qrels line {line_no}: expected at least 4 columns")
            qid, _unused, doc_id, rel = parts[:4]
            qrels[qid][doc_id] = float(rel)
    return dict(qrels)


def _load_run(path: Path) -> dict[str, list[str]]:
    run: defaultdict[str, list[tuple[int, float, str]]] = defaultdict(list)
    with _open_text(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            parts = raw_line.split()
            if len(parts) != 6:
                raise ValueError(f"run line {line_no}: expected 6 columns")
            qid, _q0, doc_id, rank, score, _tag = parts
            run[qid].append((int(rank), float(score), doc_id))
    return {
        qid: [doc_id for _rank, _score, doc_id in sorted(rows, key=lambda row: (-row[1], row[0], row[2]))]
        for qid, rows in run.items()
    }


def _dcg(relevances: list[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at(ranked_docs: list[str], rels: dict[str, float], k: int) -> float:
    gains = [rels.get(doc_id, 0.0) for doc_id in ranked_docs[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def _recall_at(ranked_docs: list[str], rels: dict[str, float], k: int) -> float:
    relevant = {doc_id for doc_id, rel in rels.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_docs[:k])
    return len(relevant & retrieved) / len(relevant)


def _rr(ranked_docs: list[str], rels: dict[str, float]) -> float:
    for idx, doc_id in enumerate(ranked_docs, start=1):
        if rels.get(doc_id, 0.0) > 0:
            return 1.0 / idx
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PAN retrieval metrics from qrels.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    args = parser.parse_args()

    qrels = _load_qrels(args.qrels)
    run = _load_run(args.run)
    qids = sorted(qrels)
    metrics = {
        "nDCG@10": [],
        "Recall@10": [],
        "Recall@100": [],
        "RR": [],
    }

    for qid in qids:
        ranked_docs = run.get(qid, [])
        rels = qrels[qid]
        metrics["nDCG@10"].append(_ndcg_at(ranked_docs, rels, 10))
        metrics["Recall@10"].append(_recall_at(ranked_docs, rels, 10))
        metrics["Recall@100"].append(_recall_at(ranked_docs, rels, 100))
        metrics["RR"].append(_rr(ranked_docs, rels))

    for name, values in metrics.items():
        mean = sum(values) / len(values) if values else 0.0
        print(f"{name}: {mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
