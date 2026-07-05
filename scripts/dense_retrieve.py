"""
Dense (bi-encoder) retrieval for source detection.

Encodes corpus and queries with sentence-transformers, builds FAISS index,
searches, and outputs a TREC run file.

Usage:
  python scripts/dense_retrieve.py \
    --corpus data/pan25_retrieval/holdout/corpus.jsonl \
    --queries data/pan25_retrieval/holdout/queries.jsonl \
    --output data/run_holdout_dense.txt \
    --model all-MiniLM-L6-v2
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Max token length for encoder (longer = slower but more context)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    # Load model
    print(f"Loading model: {args.model}...")
    t0 = time.time()
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_len
    print(f"  Loaded in {time.time() - t0:.1f}s, max_seq_length={args.max_seq_len}")

    # Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    t0 = time.time()
    doc_ids = []
    doc_texts = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(doc.get("doc_id") or doc.get("qid"))
            doc_texts.append(doc.get("default_text") or doc.get("query") or "")
    print(f"  {len(doc_ids)} docs ({time.time() - t0:.1f}s)")

    # Encode corpus
    print(f"Encoding corpus (batch_size={args.batch_size})...")
    t0 = time.time()
    doc_embeddings = model.encode(
        doc_texts, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  {doc_embeddings.shape} ({time.time() - t0:.1f}s)")

    # Build FAISS index
    print("Building FAISS index...")
    t0 = time.time()
    import faiss
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
    index.add(doc_embeddings.astype(np.float32))
    print(f"  {index.ntotal} vectors, dim={dim} ({time.time() - t0:.1f}s)")

    # Load queries
    print(f"\nLoading queries from {args.queries}...")
    t0 = time.time()
    qids = []
    qtexts = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")
    print(f"  {len(qids)} queries ({time.time() - t0:.1f}s)")

    # Encode queries
    print(f"Encoding queries...")
    t0 = time.time()
    q_embeddings = model.encode(
        qtexts, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  {q_embeddings.shape} ({time.time() - t0:.1f}s)")

    # Search
    print(f"Searching top-{args.top_k}...")
    t0 = time.time()
    scores, indices = index.search(q_embeddings.astype(np.float32), args.top_k)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # Write TREC run
    print(f"Writing {args.output}...")
    with open(args.output, "w", encoding="utf-8") as out:
        for i, qid in enumerate(qids):
            for rank, (doc_idx, score) in enumerate(zip(indices[i], scores[i]), 1):
                if score > 0:
                    out.write(f"{qid} Q0 {doc_ids[doc_idx]} {rank} {score:.6f} dense\n")

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
