"""
E5-Dense + Jaccard lexical bonus reranking.

Loads BM25 top-100 candidates, encodes queries and candidate docs with E5,
computes Dense cosine + Jaccard bonus for reranking.

Usage:
  python scripts/rerank_jaccard.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --bm25-top100 data/bm25_top100_train.jsonl \
    --output data/run_train_e5jaccard.txt
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
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--bm25-top100", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--jaccard-weight", type=float, default=2.0,
                        help="Multiplier for Jaccard bonus: score = dense * (1 + w * jaccard)")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    model.max_seq_length = 512

    # ---- Encode corpus ----
    emb_dir = Path("data/embeddings")
    emb_dir.mkdir(exist_ok=True)
    cemb_path = emb_dir / "train_e5_corpus_emb.npy"
    cids_path = emb_dir / "train_e5_doc_ids.json"
    ctok_path = emb_dir / "train_e5_doc_tokens.json"

    if cemb_path.exists() and cids_path.exists() and ctok_path.exists():
        print("Loading cached E5 corpus embeddings...")
        doc_ids = json.loads(cids_path.read_text())
        doc_emb = np.load(cemb_path)
        corpus_tokens = json.loads(ctok_path.read_text())
        corpus_tokens = {k: set(v) for k, v in corpus_tokens.items()}
        print(f"  {len(doc_ids)} docs, {doc_emb.shape}")
    else:
        print(f"Loading corpus from {args.corpus}...")
        t0 = time.time()
        doc_ids, doc_texts = [], []
        with open(args.corpus, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                doc_ids.append(d.get("doc_id") or d.get("qid"))
                doc_texts.append(d.get("default_text") or "")
        print(f"  {len(doc_ids)} docs ({time.time()-t0:.1f}s)")

        print(f"Encoding corpus with E5 (batch_size={args.batch_size})...")
        t0 = time.time()
        doc_emb = model.encode(
            ["passage: " + t for t in doc_texts],
            batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  {doc_emb.shape} ({time.time()-t0:.1f}s)")

        # Tokenize for Jaccard
        print("Tokenizing corpus for Jaccard...")
        t0 = time.time()
        corpus_tokens = {}
        for did, text in zip(doc_ids, doc_texts):
            corpus_tokens[did] = list(tokenize(text))  # list for JSON
        print(f"  {len(corpus_tokens)} docs ({time.time()-t0:.1f}s)")

        np.save(cemb_path, doc_emb)
        cids_path.write_text(json.dumps(doc_ids))
        ctok_path.write_text(json.dumps(corpus_tokens))
        # Convert back to set
        corpus_tokens = {k: set(v) for k, v in corpus_tokens.items()}
        print(f"  Saved to {emb_dir}")

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # ---- Encode queries ----
    qemb_path = emb_dir / "train_e5_query_emb.npy"
    qids_path = emb_dir / "train_e5_qids.json"

    if qemb_path.exists() and qids_path.exists():
        print("Loading cached E5 query embeddings...")
        qids = json.loads(qids_path.read_text())
        q_emb = np.load(qemb_path)
        print(f"  {len(qids)} queries, {q_emb.shape}")
    else:
        print(f"\nLoading queries from {args.queries}...")
        t0 = time.time()
        qids, qtexts = [], []
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                qids.append(q.get("qid") or q.get("query_id"))
                qtexts.append(q.get("query") or q.get("default_text") or "")
        print(f"  {len(qids)} queries ({time.time()-t0:.1f}s)")

        print(f"Encoding queries with E5...")
        t0 = time.time()
        q_emb = model.encode(
            ["query: " + t for t in qtexts],
            batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  {q_emb.shape} ({time.time()-t0:.1f}s)")
        np.save(qemb_path, q_emb)
        qids_path.write_text(json.dumps(qids))
        print(f"  Saved to {emb_dir}")

    qid_to_idx = {qid: i for i, qid in enumerate(qids)}

    # ---- Pre-tokenize queries ----
    print("\nPre-tokenizing queries for Jaccard...")
    t0 = time.time()
    query_tokens = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            q_text = q.get("query") or q.get("default_text") or ""
            query_tokens[qid] = tokenize(q_text)
    print(f"  {len(query_tokens)} queries ({time.time()-t0:.1f}s)")

    # ---- Load BM25 top-100 ----
    print(f"Loading BM25 top-100 from {args.bm25_top100}...")
    t0 = time.time()
    candidates = {}
    with open(args.bm25_top100, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            candidates[q["qid"]] = [(c["doc_id"], c["bm25_score"]) for c in q["candidates"]]
    print(f"  {len(candidates)} queries ({time.time()-t0:.1f}s)")

    # ---- Rerank: Dense cosine + Jaccard ----
    print(f"\nReranking with E5 + Jaccard (w={args.jaccard_weight})...")
    t0 = time.time()
    w = args.jaccard_weight
    total_pairs = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for qi, qid in enumerate(qids):
            if qid not in candidates or qid not in qid_to_idx:
                continue

            q_idx = qid_to_idx[qid]
            q_vec = q_emb[q_idx]
            q_tokens = query_tokens.get(qid, set())

            scored = []
            for doc_id, bm25_score in candidates[qid]:
                if doc_id not in doc_id_to_idx:
                    continue
                total_pairs += 1

                d_idx = doc_id_to_idx[doc_id]
                dense_score = float(q_vec @ doc_emb[d_idx])

                d_tokens = corpus_tokens.get(doc_id, set())
                if q_tokens and d_tokens:
                    jaccard = len(q_tokens & d_tokens) / max(len(q_tokens | d_tokens), 1)
                else:
                    jaccard = 0.0

                final_score = dense_score * (1.0 + w * jaccard)
                scored.append((doc_id, final_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            for rank, (doc_id, final_score) in enumerate(scored[:10], 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {final_score:.6f} e5jaccard\n")

            if (qi + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {qi+1}/{len(qids)} ({elapsed:.1f}s, {total_pairs/elapsed:.0f} pairs/s)")

    elapsed = time.time() - t0
    print(f"Done: {len(qids)} queries in {elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
