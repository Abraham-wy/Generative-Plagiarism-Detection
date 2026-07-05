"""
Local evaluation pipeline for generative plagiarism detection (source retrieval).

Evaluates a retrieval run against qrels using standard IR metrics.

Usage:
  python scripts/evaluate.py --run run.txt --qrels qrels.txt
  python scripts/evaluate.py --run data/my-run.txt --qrels data/pan25_retrieval/train/qrels.txt

Outputs:
  - nDCG@1,3,5,10
  - Recall@1,3,5,10
  - MRR
  - MAP
"""

import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_qrels(path):
    """Load TREC qrels: qid 0 doc_id relevance"""
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][doc_id] = rel
    return qrels


def load_run(path):
    """Load TREC run: qid Q0 doc_id rank score tag"""
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score, _ = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4]), parts[5]
            run[qid].append((doc_id, score))
    # Sort by score descending
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run


def dcg_at_k(rels, k):
    """DCG@k"""
    dcg = 0.0
    for i in range(min(k, len(rels))):
        rel = rels[i] if i < len(rels) else 0
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    return dcg


def ndcg_at_k(run, qrels, k):
    """nDCG@k averaged over all queries"""
    scores = []
    for qid in qrels:
        if qid not in run:
            scores.append(0.0)
            continue
        ranked_docs = [doc_id for doc_id, _ in run[qid][:k]]
        rels = [qrels[qid].get(doc_id, 0) for doc_id in ranked_docs]
        dcg = dcg_at_k(rels, k)

        # Ideal DCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)
        idcg = dcg_at_k(ideal_rels, k)
        if idcg == 0:
            scores.append(0.0)
        else:
            scores.append(dcg / idcg)
    return np.mean(scores) if scores else 0.0


def recall_at_k(run, qrels, k):
    """Recall@k"""
    scores = []
    for qid in qrels:
        total_rel = sum(1 for v in qrels[qid].values() if v > 0)
        if total_rel == 0:
            continue
        if qid not in run:
            scores.append(0.0)
            continue
        ranked_docs = [doc_id for doc_id, _ in run[qid][:k]]
        found = sum(1 for doc_id in ranked_docs if qrels[qid].get(doc_id, 0) > 0)
        scores.append(found / total_rel)
    return np.mean(scores) if scores else 0.0


def mrr(run, qrels):
    """Mean Reciprocal Rank"""
    scores = []
    for qid in qrels:
        if qid not in run:
            scores.append(0.0)
            continue
        for rank, (doc_id, _) in enumerate(run[qid], 1):
            if qrels[qid].get(doc_id, 0) > 0:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return np.mean(scores) if scores else 0.0


def map_score(run, qrels):
    """Mean Average Precision"""
    scores = []
    for qid in qrels:
        total_rel = sum(1 for v in qrels[qid].values() if v > 0)
        if total_rel == 0:
            continue
        if qid not in run:
            scores.append(0.0)
            continue
        ap = 0.0
        found = 0
        for rank, (doc_id, _) in enumerate(run[qid], 1):
            if qrels[qid].get(doc_id, 0) > 0:
                found += 1
                ap += found / rank
        ap /= total_rel
        scores.append(ap)
    return np.mean(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval run")
    parser.add_argument("--run", type=Path, required=True, help="TREC run file")
    parser.add_argument("--qrels", type=Path, required=True, help="TREC qrels file")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    print(f"Queries in qrels: {len(qrels)}")
    print(f"Queries in run: {len(run)}")
    print(f"Common queries: {len(set(qrels) & set(run))}")
    print()

    ks = [1, 3, 5, 10, 20]
    print(f"{'Metric':<14}", end="")
    for k in ks:
        print(f"@{k:<5}", end="")
    print()

    for metric_name, func in [("nDCG", None), ("Recall", None)]:
        print(f"{metric_name:<14}", end="")
        for k in ks:
            if metric_name == "nDCG":
                v = ndcg_at_k(run, qrels, k)
            else:
                v = recall_at_k(run, qrels, k)
            print(f"{v:<6.4f}", end="")
        print()

    m = mrr(run, qrels)
    m_ap = map_score(run, qrels)
    print(f"\nMRR:  {m:.4f}")
    print(f"MAP:  {m_ap:.4f}")


if __name__ == "__main__":
    main()
