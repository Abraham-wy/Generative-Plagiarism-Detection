"""
Dense encoding + retrieval + hybrid fusion for full train set.

Saves embeddings to disk to avoid re-encoding.

Usage:
  python scripts/dense_encode.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --output data/run_train_hybrid.txt \
    --model all-MiniLM-L6-v2
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def load_jsonl(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids.append(d.get("doc_id") or d.get("qid") or d.get("query_id"))
            texts.append(d.get("default_text") or d.get("query") or "")
    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_len

    # E5-style models need query/passage prefixes
    is_e5 = 'e5' in args.model.lower()
    query_prefix = "query: " if is_e5 else ""
    doc_prefix = "passage: " if is_e5 else ""

    # ---- Encode corpus ----
    emb_dir = Path("data/embeddings")
    emb_dir.mkdir(exist_ok=True)
    corpus_emb_path = emb_dir / "train_corpus_emb.npy"
    doc_ids_path = emb_dir / "train_doc_ids.json"

    if corpus_emb_path.exists() and doc_ids_path.exists():
        print("Loading cached corpus embeddings...")
        doc_ids = json.loads(doc_ids_path.read_text())
        doc_emb = np.load(corpus_emb_path)
        print(f"  {len(doc_ids)} docs, {doc_emb.shape}")
    else:
        print(f"Loading corpus from {args.corpus}...")
        t0 = time.time()
        doc_ids, doc_texts = load_jsonl(args.corpus)
        print(f"  {len(doc_ids)} docs ({time.time() - t0:.1f}s)")

        print(f"Encoding corpus (batch_size={args.batch_size})...")
        t0 = time.time()
        doc_emb = model.encode(
            [doc_prefix + t for t in doc_texts], batch_size=args.batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        )
        print(f"  {doc_emb.shape} ({time.time() - t0:.1f}s)")

        np.save(corpus_emb_path, doc_emb)
        doc_ids_path.write_text(json.dumps(doc_ids))
        print(f"  Saved to {emb_dir}")

    # ---- Encode queries ----
    query_emb_path = emb_dir / "train_query_emb.npy"
    qids_path = emb_dir / "train_qids.json"

    if query_emb_path.exists() and qids_path.exists():
        print("Loading cached query embeddings...")
        qids = json.loads(qids_path.read_text())
        q_emb = np.load(query_emb_path)
        print(f"  {len(qids)} queries, {q_emb.shape}")
    else:
        print(f"\nLoading queries from {args.queries}...")
        t0 = time.time()
        qids, qtexts = load_jsonl(args.queries)
        print(f"  {len(qids)} queries ({time.time() - t0:.1f}s)")

        print(f"Encoding queries...")
        t0 = time.time()
        q_emb = model.encode(
            [query_prefix + t for t in qtexts], batch_size=args.batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        )
        print(f"  {q_emb.shape} ({time.time() - t0:.1f}s)")

        np.save(query_emb_path, q_emb)
        qids_path.write_text(json.dumps(qids))
        print(f"  Saved to {emb_dir}")

    # ---- Dense search + BM25 hybrid ----
    print(f"\nLoading BM25 top-100...")
    t0 = time.time()
    bm25_scores = {}
    bm25_candidates = {}
    bm25_top100_path = Path("data/bm25_top100_train.jsonl")
    if bm25_top100_path.exists():
        with open(bm25_top100_path) as f:
            for line in f:
                q = json.loads(line.strip())
                qid = q["qid"]
                bm25_candidates[qid] = {c["doc_id"]: c["bm25_score"] for c in q["candidates"]}
    print(f"  {len(bm25_candidates)} queries ({time.time() - t0:.1f}s)")

    # Map doc_ids to indices for FAISS-less search
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    print(f"Searching top-{args.top_k} (numpy batched)...")
    t0 = time.time()
    dense_ranks = {}  # qid -> {doc_id: rank}
    batch = 200  # process queries in batches to avoid OOM

    for i in range(0, len(q_emb), batch):
        batch_end = min(i + batch, len(q_emb))
        sims = q_emb[i:batch_end] @ doc_emb.T  # (batch, num_docs)
        for j in range(batch_end - i):
            q_idx = i + j
            qid = qids[q_idx]
            s = sims[j]
            top_idx = np.argpartition(s, -args.top_k)[-args.top_k:]
            top_idx = top_idx[np.argsort(s[top_idx])[::-1]]
            dense_ranks[qid] = {}
            for rank, doc_idx in enumerate(top_idx, 1):
                did = doc_ids[doc_idx]
                dense_ranks[qid][did] = rank
        if (i + batch) % 2000 == 0:
            print(f"  {batch_end}/{len(q_emb)} ({time.time() - t0:.1f}s)")

    print(f"  Search done ({time.time() - t0:.1f}s)")

    # ---- RRF fusion ----
    print(f"\nFusing with BM25 (RRF K={args.rrf_k})...")
    K = args.rrf_k
    default_rank = 200

    with open(args.output, "w", encoding="utf-8") as out:
        for qid in bm25_candidates:
            if qid not in dense_ranks:
                continue

            candidates = {}
            # From BM25
            for doc_id, bm25_score in bm25_candidates[qid].items():
                bm25_r = 1  # approximate, BM25 top-100 are the candidates
                dense_r = dense_ranks[qid].get(doc_id, default_rank)
                # Sort BM25 candidates by score to get approximate rank
                candidates[doc_id] = (bm25_score, dense_r)

            # From Dense (not already in BM25)
            for doc_id, dense_r in dense_ranks[qid].items():
                if doc_id not in candidates:
                    bm25_score = 0.0
                    candidates[doc_id] = (bm25_score, dense_r)

            # Build proper BM25 ranks
            sorted_bm25 = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            bm25_rank_map = {}
            for rank, (doc_id, _) in enumerate(sorted_bm25, 1):
                bm25_rank_map[doc_id] = rank

            # Compute RRF
            rrf_scores = {}
            for doc_id, (_, dense_r) in candidates.items():
                bm25_r = bm25_rank_map.get(doc_id, default_rank)
                rrf_scores[doc_id] = 1/(K + bm25_r) + 1/(K + dense_r)

            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (doc_id, score) in enumerate(ranked, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} hybrid\n")

    print(f"Written to {args.output}")

    # Quick eval
    print("\n=== Quick eval ===")
    qrels = {}
    qrels_path = Path("data/pan25_retrieval/train/qrels.txt")
    with open(qrels_path) as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0:
                qrels[qid] = doc

    run = {}
    with open(args.output) as f:
        for line in f:
            parts = line.strip().split()
            qid, doc = parts[0], parts[2]
            if qid not in run:
                run[qid] = []
            run[qid].append(doc)

    hits1 = 0; hits10 = 0; mrr_sum = 0
    for qid in qrels:
        if qid not in run:
            continue
        rel = qrels[qid]
        top10 = run[qid][:10]
        if top10 and top10[0] == rel:
            hits1 += 1
        if rel in top10:
            hits10 += 1
            for r, d in enumerate(top10, 1):
                if d == rel:
                    mrr_sum += 1.0 / r
                    break
    n = len(qrels)
    print(f"Queries: {n}")
    print(f"MRR: {mrr_sum/n:.4f}  nDCG@1: {hits1/n:.4f}  Recall@10: {hits10/n:.4f}")


if __name__ == "__main__":
    main()
