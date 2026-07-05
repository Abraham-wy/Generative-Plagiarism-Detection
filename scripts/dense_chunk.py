"""
Chunking-based Dense retrieval.

Splits documents into overlapping chunks (256 tokens, 128 stride),
encodes all chunks, then scores queries as max(chunk_similarity).

Usage:
  python scripts/dense_chunk.py \
    --corpus data/pan25_retrieval/holdout/corpus.jsonl \
    --queries data/pan25_retrieval/holdout/queries.jsonl \
    --output data/run_holdout_chunk.txt \
    --model intfloat/e5-base-v2
"""

import argparse
import json
import re
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def chunk_text(text, chunk_tokens=256, stride_tokens=128):
    """
    Split text into overlapping chunks of ~chunk_tokens tokens.
    Returns list of chunk texts (no char offsets needed for retrieval).
    """
    tokens = tokenize(text)
    if len(tokens) <= chunk_tokens:
        return [text[:3000]]  # Cap single chunk

    chunks = []
    i = 0
    while i < len(tokens):
        end = min(i + chunk_tokens, len(tokens))
        chunks.append(' '.join(tokens[i:end]))
        i += stride_tokens

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--stride-tokens", type=int, default=128)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max docs to encode (0=all, for quick testing)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max queries to search (0=all)")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    model.max_seq_length = 256

    print(f"Model: {args.model}")
    print(f"Chunking: {args.chunk_tokens} tokens, {args.stride_tokens} stride")

    # ---- Chunk + Encode corpus ----
    print(f"\nLoading corpus from {args.corpus}...")
    t0 = time.time()
    all_chunks = []  # list of (doc_id, chunk_text)
    doc_ids = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("qid")
            text = doc.get("default_text") or doc.get("query") or ""

            chunks = chunk_text(text, args.chunk_tokens, args.stride_tokens)
            doc_ids.append(doc_id)
            for ct in chunks:
                all_chunks.append((doc_id, ct))

            if args.max_docs > 0 and len(doc_ids) >= args.max_docs:
                break

    print(f"  {len(doc_ids)} docs → {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")
    print(f"  Avg chunks/doc: {len(all_chunks)/max(len(doc_ids),1):.1f}")

    # Encode all chunks with passage prefix
    print(f"Encoding {len(all_chunks)} chunks (batch_size={args.batch_size})...")
    t0 = time.time()
    chunk_texts = ["passage: " + c[1] for c in all_chunks]
    chunk_emb = model.encode(chunk_texts, batch_size=args.batch_size,
                             show_progress_bar=True, normalize_embeddings=True)
    print(f"  {chunk_emb.shape} ({time.time()-t0:.1f}s)")

    # Build doc_id → [chunk indices] mapping
    doc_to_chunks = defaultdict(list)
    for ci, (doc_id, _) in enumerate(all_chunks):
        doc_to_chunks[doc_id].append(ci)

    # ---- Encode queries ----
    print(f"\nLoading queries from {args.queries}...")
    t0 = time.time()
    qids, qtexts = [], []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append("query: " + (q.get("query") or q.get("default_text") or ""))
            if args.max_queries > 0 and len(qids) >= args.max_queries:
                break
    print(f"  {len(qids)} queries ({time.time()-t0:.1f}s)")

    print(f"Encoding {len(qids)} queries...")
    t0 = time.time()
    q_emb = model.encode(qtexts, batch_size=args.batch_size,
                         show_progress_bar=True, normalize_embeddings=True)
    print(f"  {q_emb.shape} ({time.time()-t0:.1f}s)")

    # ---- Search: max chunk similarity per document ----
    print(f"\nSearching top-{args.top_k} (max chunk similarity per doc)...")
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for qi in range(len(qids)):
            q_vec = q_emb[qi:qi+1]  # (1, dim)

            # Score all chunks: (1, dim) @ (dim, n_chunks) → (1, n_chunks)
            # For efficiency, do in batches to avoid OOM
            doc_scores = {}
            for doc_id, chunk_indices in doc_to_chunks.items():
                if not chunk_indices:
                    continue
                chunk_sims = q_vec @ chunk_emb[chunk_indices].T  # (1, n)
                doc_scores[doc_id] = float(chunk_sims.max())

            # Top-k by max chunk score
            top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
            for rank, (doc_id, score) in enumerate(top_docs, 1):
                out.write(f"{qids[qi]} Q0 {doc_id} {rank} {score:.6f} chunk\n")

            if (qi + 1) % 100 == 0:
                print(f"  {qi+1}/{len(qids)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(qids)} queries in {elapsed:.1f}s ({len(qids)/elapsed:.0f} q/s)")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
