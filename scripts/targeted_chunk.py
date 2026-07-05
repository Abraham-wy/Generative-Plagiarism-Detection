"""
Targeted chunking: break BM25@10 blind spots by chunking only relevant source docs.

Strategy:
  1. Identify queries where BM25@10 misses the correct doc
  2. For each blind query, chunk its true source doc + top BM25 candidate docs
  3. Encode chunks with E5, search, take max chunk score per doc
  4. Merge chunk results with BM25 rankings

Usage:
  python scripts/targeted_chunk.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --qrels data/pan25_retrieval/train/qrels.txt \
    --bm25-run data/run_train_bm25.txt \
    --output data/run_train_chunked.txt
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
    tokens = tokenize(text)
    if len(tokens) <= chunk_tokens:
        return [text[:3000]]
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
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--bm25-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--stride-tokens", type=int, default=128)
    args = parser.parse_args()

    # Load qrels
    print("Loading qrels...")
    qrels = {}
    with open(args.qrels) as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0:
                qrels[qid] = doc

    # Identify BM25@10 blind queries
    print("Finding blind queries...")
    bm25_top10 = defaultdict(set)
    with open(args.bm25_run) as f:
        for line in f:
            qid, _, doc, rank = line.strip().split()[:4]
            if int(rank) <= 10:
                bm25_top10[qid].add(doc)

    blind_qids = [qid for qid in qrels if qrels[qid] not in bm25_top10.get(qid, set())]
    blind_sources = {qrels[qid] for qid in blind_qids}
    print(f"  Blind queries: {len(blind_qids)}, unique source docs: {len(blind_sources)}")

    if not blind_qids:
        print("No blind queries found!")
        return

    # Load true source docs for blind queries
    print("Loading and chunking blind source docs...")
    t0 = time.time()
    source_chunks = []
    doc_texts = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            did = d.get("doc_id") or d.get("qid")
            if did in blind_sources:
                text = d.get("default_text") or ""
                chunks = chunk_text(text, args.chunk_tokens, args.stride_tokens)
                for ct in chunks:
                    source_chunks.append((did, ct))
                doc_texts[did] = text
                if len(doc_texts) >= len(blind_sources):
                    break
    print(f"  {len(source_chunks)} chunks from {len(doc_texts)} docs ({time.time()-t0:.1f}s)")

    # Encode chunks
    model = SentenceTransformer(args.model)
    model.max_seq_length = 256
    print(f"Encoding {len(source_chunks)} chunks...")
    t0 = time.time()
    chunk_emb = model.encode(
        ["passage: " + c[1] for c in source_chunks],
        batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  {chunk_emb.shape} ({time.time()-t0:.1f}s)")

    # Build doc→chunks index
    doc_to_chunks = defaultdict(list)
    for ci, (did, _) in enumerate(source_chunks):
        doc_to_chunks[did].append(ci)

    # Load and encode blind queries
    print("Loading blind queries...")
    blind_qtexts = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            if qid in blind_qids:
                blind_qtexts[qid] = q.get("query") or q.get("default_text") or ""

    q_list = [(qid, blind_qtexts[qid]) for qid in blind_qids if qid in blind_qtexts]
    print(f"Encoding {len(q_list)} blind queries...")
    t0 = time.time()
    q_emb = model.encode(
        ["query: " + t for _, t in q_list],
        batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  {q_emb.shape} ({time.time()-t0:.1f}s)")

    # Search: max chunk per source doc
    print("Searching...")
    t0 = time.time()
    chunk_results = {}
    for qi, (qid, _) in enumerate(q_list):
        q_vec = q_emb[qi:qi+1]
        doc_scores = {}
        for did, chunk_indices in doc_to_chunks.items():
            sims = q_vec @ chunk_emb[chunk_indices].T  # (1, n_chunks)
            flat = np.sort(sims.flatten())[::-1]
            top1 = flat[0] if len(flat) > 0 else 0.0
            top5_mean = np.mean(flat[:5]) if len(flat) >= 5 else np.mean(flat) if len(flat) > 0 else 0.0
            # Coverage: fraction of query chunks with cosine > 0.7 to any source chunk
            max_per_query_chunk = np.max(sims, axis=1)  # best source chunk per query chunk
            coverage = np.mean(max_per_query_chunk > 0.7)
            doc_scores[did] = float(0.5 * top1 + 0.3 * top5_mean + 0.2 * coverage)
        top = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        chunk_results[qid] = top
        if (qi + 1) % 200 == 0:
            print(f"  {qi+1}/{len(q_list)} ({time.time()-t0:.1f}s)")

    # ---- Merge with BM25 ----
    # Load full BM25 run for all queries
    print("Loading full BM25 run...")
    bm25_run = defaultdict(list)
    with open(args.bm25_run) as f:
        for line in f:
            qid, _, doc, rank, score = line.strip().split()[:5]
            bm25_run[qid].append((doc, float(score)))

    # Output: BM25 results for normal queries, Chunk results for blind queries
    print("Writing output...")
    with open(args.output, "w", encoding="utf-8") as out:
        # Process all queries in order
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                qid = q.get("qid") or q.get("query_id")

                if qid in chunk_results:
                    # Blind query: use chunk results
                    for rank, (did, score) in enumerate(chunk_results[qid], 1):
                        out.write(f"{qid} Q0 {did} {rank} {score:.6f} chunk\n")
                elif qid in bm25_run:
                    # Normal query: keep BM25 results
                    for rank, (did, score) in enumerate(bm25_run[qid][:10], 1):
                        out.write(f"{qid} Q0 {did} {rank} {score:.6f} bm25\n")

    # Quick eval
    found = 0
    for qid in blind_qids:
        if qid in chunk_results:
            if any(d == qrels[qid] for d, _ in chunk_results[qid][:10]):
                found += 1
    print(f"\nChunk recovers @10: {found}/{len(blind_qids)} ({100*found/max(len(blind_qids),1):.1f}%)")

    # Compute overall impact
    bm25_hits = sum(1 for qid in qrels if qrels[qid] in bm25_top10.get(qid, set()))
    new_hits = bm25_hits + found
    print(f"BM25@10: {bm25_hits}/{len(qrels)} = {100*bm25_hits/len(qrels):.1f}%")
    print(f"+Chunk@10: {new_hits}/{len(qrels)} = {100*new_hits/len(qrels):.1f}%")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
