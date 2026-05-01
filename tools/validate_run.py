#!/usr/bin/env python3
"""
Validate a PAN/TREC run file.

Checks:
- gzip/plain text readability
- exactly six columns per line
- Q0 marker
- numeric rank and score
- at most N results per query
- non-increasing score order per query
- optional contiguous ranks starting at 0
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter, defaultdict
from pathlib import Path


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_run(path: Path, max_per_query: int = 1000, require_contiguous_ranks: bool = True) -> list[str]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    previous_score: dict[str, float] = {}
    ranks: defaultdict[str, list[int]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    with _open_text(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                errors.append(f"line {line_no}: empty line")
                continue

            parts = line.split()
            if len(parts) != 6:
                errors.append(f"line {line_no}: expected 6 columns, got {len(parts)}")
                continue

            qid, q0, doc_id, rank_raw, score_raw, _tag = parts
            if q0 != "Q0":
                errors.append(f"line {line_no}: second column must be Q0, got {q0!r}")

            try:
                rank = int(rank_raw)
            except ValueError:
                errors.append(f"line {line_no}: rank is not an integer: {rank_raw!r}")
                continue

            try:
                score = float(score_raw)
            except ValueError:
                errors.append(f"line {line_no}: score is not a float: {score_raw!r}")
                continue

            pair = (qid, doc_id)
            if pair in seen_pairs:
                errors.append(f"line {line_no}: duplicate doc {doc_id!r} for qid {qid!r}")
            seen_pairs.add(pair)

            counts[qid] += 1
            ranks[qid].append(rank)
            if counts[qid] > max_per_query:
                errors.append(f"line {line_no}: qid {qid!r} exceeds {max_per_query} results")

            if qid in previous_score and score > previous_score[qid]:
                errors.append(
                    f"line {line_no}: score for qid {qid!r} increases "
                    f"from {previous_score[qid]:.12g} to {score:.12g}"
                )
            previous_score[qid] = score

    if require_contiguous_ranks:
        for qid, q_ranks in ranks.items():
            expected = list(range(len(q_ranks)))
            if q_ranks != expected:
                errors.append(
                    f"qid {qid!r}: ranks must be contiguous from 0; "
                    f"got first ranks {q_ranks[:10]!r}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a PAN 2026 TREC run file.")
    parser.add_argument("run", type=Path, help="Path to run.txt or run.txt.gz.")
    parser.add_argument("--max-per-query", type=int, default=1000)
    parser.add_argument("--allow-noncontiguous-ranks", action="store_true")
    args = parser.parse_args()

    errors = validate_run(
        args.run,
        max_per_query=args.max_per_query,
        require_contiguous_ranks=not args.allow_noncontiguous_ranks,
    )
    if errors:
        print(f"INVALID: {args.run}")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more errors")
        return 1

    print(f"OK: {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
