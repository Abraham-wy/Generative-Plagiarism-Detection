#!/usr/bin/env python3
"""
Run a small retrieval parameter grid and record metadata.

This script is intentionally conservative: by default it runs BM25+RRF only
because that is the fastest way to test recall-oriented settings. Use
--include-rerank to add dense re-ranking model variants.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _count_run_lines(run_path: Path) -> int:
    import gzip

    with gzip.open(run_path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PAN retrieval experiment grid.")
    parser.add_argument("--dataset", default="test-data", help="ir_datasets ID or local data directory.")
    parser.add_argument("--index-root", type=Path, default=Path("/tmp/pan26-indexes"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--n-sub-queries", default="3,5,8")
    parser.add_argument("--sub-query-tokens", default="64,128")
    parser.add_argument("--bm25-top-k", default="200,500,1000")
    parser.add_argument("--final-top-k", type=int, default=1000)
    parser.add_argument("--include-rerank", action="store_true")
    parser.add_argument("--rerank-models", default="all-MiniLM-L6-v2,multi-qa-mpnet-base-dot-v1")
    parser.add_argument("--clean", action="store_true", help="Delete each run output before re-running it.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    retrieve_py = repo_root / "retrieve.py"
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    rerank_modes: list[tuple[bool, str | None]] = [(False, None)]
    if args.include_rerank:
        rerank_modes.extend((True, model) for model in _csv_strings(args.rerank_models))

    rows: list[dict[str, object]] = []
    grid = itertools.product(
        _csv_ints(args.n_sub_queries),
        _csv_ints(args.sub_query_tokens),
        _csv_ints(args.bm25_top_k),
        rerank_modes,
    )

    for n_sub, sub_tokens, bm25_top_k, (rerank, rerank_model) in grid:
        mode = "rerank" if rerank else "bm25rrf"
        model_slug = (rerank_model or "none").replace("/", "__")
        run_name = f"{mode}_n{n_sub}_tok{sub_tokens}_bm25{bm25_top_k}_{model_slug}"
        run_dir = output_root / run_name
        index_dir = args.index_root / "shared"

        if args.clean and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            str(retrieve_py),
            "--dataset",
            args.dataset,
            "--output",
            str(run_dir),
            "--index",
            str(index_dir),
            "--n-sub-queries",
            str(n_sub),
            "--sub-query-tokens",
            str(sub_tokens),
            "--bm25-top-k",
            str(bm25_top_k),
            "--final-top-k",
            str(args.final_top_k),
            "--tag",
            run_name[:64],
            "--force",
        ]
        if rerank:
            cmd.extend(["--rerank", "--rerank-model", rerank_model or "all-MiniLM-L6-v2"])
        else:
            cmd.append("--no-rerank")

        started = time.time()
        completed = subprocess.run(cmd, cwd=repo_root)
        elapsed = round(time.time() - started, 2)
        run_path = run_dir / "run.txt.gz"
        line_count = _count_run_lines(run_path) if run_path.exists() else 0

        metadata = {
            "name": run_name,
            "dataset": args.dataset,
            "n_sub_queries": n_sub,
            "sub_query_tokens": sub_tokens,
            "bm25_top_k": bm25_top_k,
            "final_top_k": args.final_top_k,
            "rerank": rerank,
            "rerank_model": rerank_model or "",
            "output": str(run_path),
            "elapsed_seconds": elapsed,
            "line_count": line_count,
            "return_code": completed.returncode,
            "command": " ".join(cmd),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows.append(metadata)

        if completed.returncode != 0:
            print(f"FAILED: {run_name}", file=sys.stderr)
            break

    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {manifest_path}")

    return 0 if all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
