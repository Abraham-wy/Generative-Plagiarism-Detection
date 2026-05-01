#!/usr/bin/env python3
"""
Create lightweight inspection reports for a run file.

The report helps manual error analysis before qrels are available: top-k docs per
query, score gaps, repeated documents, and query/result counts.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_run(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            qid, _q0, doc_id, rank, score, tag = raw_line.split()
            yield {
                "qid": qid,
                "doc_id": doc_id,
                "rank": int(rank),
                "score": float(score),
                "tag": tag,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a PAN run for manual inspection.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    by_qid: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    doc_counts: Counter[str] = Counter()
    for row in _read_run(args.run):
        by_qid[str(row["qid"])].append(row)
        doc_counts[str(row["doc_id"])] += 1

    report = {
        "run": str(args.run),
        "num_queries": len(by_qid),
        "num_rows": sum(len(rows) for rows in by_qid.values()),
        "results_per_query": {qid: len(rows) for qid, rows in sorted(by_qid.items())},
        "most_repeated_docs": doc_counts.most_common(20),
        "top_results": {},
    }

    for qid, rows in sorted(by_qid.items()):
        rows = sorted(rows, key=lambda row: row["rank"])
        top_rows = rows[: args.top_k]
        top_scores = [float(row["score"]) for row in top_rows]
        gaps = [
            round(top_scores[i] - top_scores[i + 1], 8)
            for i in range(len(top_scores) - 1)
        ]
        report["top_results"][qid] = {
            "top_docs": top_rows,
            "score_gaps": gaps,
        }

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
