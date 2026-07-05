"""
Cross-encoder reranking of BM25 top-100 candidates.

Outputs a TREC run file with CE-rescored rankings.

Usage:
  python scripts/ce_rerank.py \
    --input data/bm25_top100_holdout.jsonl \
    --output data/run_holdout_ce.txt \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path
from sentence_transformers import CrossEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K candidates to keep after reranking")
    args = parser.parse_args()

    # Load candidates
    print(f"Loading candidates from {args.input}...")
    queries = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    print(f"  {len(queries)} queries")

    # Load model
    print(f"Loading model: {args.model}...")
    t0 = time.time()
    model = CrossEncoder(args.model, max_length=512)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Rerank each query
    print(f"Reranking (batch_size={args.batch_size})...")
    t0 = time.time()
    total_pairs = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for i, q in enumerate(queries):
            qid = q["qid"]
            query_text = q["query_text"]
            candidates = q["candidates"]

            if not candidates:
                continue

            # Prepare (query, doc) pairs
            pairs = [(query_text, c["text"]) for c in candidates]
            total_pairs += len(pairs)

            # Score
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)

            # Pair with candidates and sort by CE score descending
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True,
            )[:args.top_k]

            # Write TREC format
            for rank, (cand, ce_score) in enumerate(ranked, 1):
                out.write(
                    f"{qid} Q0 {cand['doc_id']} {rank} {ce_score:.6f} ce\n"
                )

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(queries)} ({elapsed:.1f}s, {total_pairs/elapsed:.0f} pairs/s)")

    elapsed = time.time() - t0
    print(f"Done: {len(queries)} queries, {total_pairs} pairs in {elapsed:.1f}s "
          f"({total_pairs/elapsed:.0f} pairs/s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
