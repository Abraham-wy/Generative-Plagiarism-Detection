"""
Export BM25 top-100 candidates for cross-encoder reranking.

Builds inverted index, retrieves top-100 per query, attaches doc text snippets,
outputs JSONL ready for sentence-transformers CrossEncoder.

Usage:
  python scripts/export_bm25_top100.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --output data/runs/bm25_top100_train.jsonl \
    --query-max-chars 2000

Output format (one JSON object per line):
  {"qid": "q-xxx", "query_text": "...", "candidates": [
    {"doc_id": "d-xxx", "bm25_score": 123.4, "text": "first 1500 chars"},
    ...
  ]}
"""

import argparse
import json
import time
from pathlib import Path

# Import BM25 from sibling script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_bm25 import BM25, load_queries, _stream_corpus

SNIPPET_CHARS = 1500


def main():
    parser = argparse.ArgumentParser(description="Export BM25 top-100 for reranking")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-df-ratio", type=float, default=0.3)
    parser.add_argument("--query-max-chars", type=int, default=2000)
    parser.add_argument("--snippet-chars", type=int, default=SNIPPET_CHARS)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Build index ---
    t0 = time.time()
    print("Building BM25 index...")
    bm25 = BM25(k1=args.k1, b=args.b)
    bm25.index(str(args.corpus), max_df_ratio=args.max_df_ratio)
    print(f"Index done ({time.time() - t0:.1f}s)")

    # --- Load queries ---
    t0 = time.time()
    print(f"Loading queries...")
    queries = list(load_queries(args.queries))
    print(f"Loaded {len(queries)} queries ({time.time() - t0:.1f}s)")

    # --- Build doc_id -> text map for snippets ---
    t0 = time.time()
    print("Loading corpus for text snippets...")
    doc_texts = {}
    for doc_id_str, text in _stream_corpus(args.corpus):
        doc_texts[doc_id_str] = text[:args.snippet_chars]
    print(f"Loaded {len(doc_texts)} doc snippets ({time.time() - t0:.1f}s)")

    # --- Retrieve and export ---
    qmax = args.query_max_chars
    print(f"Retrieving top-{args.top_k} and exporting...")
    t0 = time.time()
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, (qid, full_qtext) in enumerate(queries):
            qtext = full_qtext[:qmax] if qmax > 0 else full_qtext
            results = bm25.search(qtext, top_k=args.top_k)

            candidates = []
            for doc_id, score in results:
                candidates.append({
                    "doc_id": doc_id,
                    "bm25_score": score,
                    "text": doc_texts.get(doc_id, ""),
                })

            record = {
                "qid": qid,
                "query_text": full_qtext[:qmax] if qmax > 0 else full_qtext,
                "candidates": candidates,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(queries)} ({len(queries)/elapsed:.0f} q/s)")

    elapsed = time.time() - t0
    print(f"Done: {written} queries in {elapsed:.1f}s ({written/elapsed:.0f} q/s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
