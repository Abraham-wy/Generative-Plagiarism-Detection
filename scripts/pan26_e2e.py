"""
PAN26 end-to-end verification: source retrieval + text alignment on spot_check.

Outputs the 4 key metrics for retrieval, and alignment F1.

Usage:
  python scripts/pan26_e2e.py --split spot_check
"""

import argparse, json, re, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bm25_index import InvertedIndexBM25
from scripts.align_v2 import AlignerV2, build_truth_from_xml, sent_tokenize


def load_jsonl(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            data[d.get("doc_id") or d.get("qid") or d.get("query_id")] = \
                d.get("default_text") or d.get("query") or ""
    return data


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0:
                qrels[qid] = doc
    return qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="spot_check",
                        choices=["spot_check", "holdout"])
    parser.add_argument("--threshold", type=float, default=0.89)
    parser.add_argument("--top-k-sources", type=int, default=1)
    args = parser.parse_args()

    base = Path("data/pan25_retrieval") / args.split
    truth_dir = Path("pan25") / (
        "00_spot_check/00_spot_check_truth" if args.split == "spot_check"
        else "02_validation/02_validation_truth"
    )

    print(f"=== PAN26 E2E: {args.split} ===")

    # Load data
    print("\nLoading data...")
    corpus = load_jsonl(base / "corpus.jsonl")
    suspicious = load_jsonl(base / "queries.jsonl")
    qrels = load_qrels(base / "qrels.txt")

    # Load ID mapping
    id2file, file2id = {}, {}
    mapping_path = base / "id_mapping.tsv"
    if mapping_path.exists():
        with open(mapping_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    id2file[parts[2]] = parts[3]
                    file2id[parts[3]] = parts[2]
        print(f"  ID mapping: {len(id2file)} entries")

    # ---- Phase 1: Source Retrieval ----
    print("\n--- Phase 1: Source Retrieval ---")

    # Build BM25 index
    idx = InvertedIndexBM25()
    corp_jsonl = base / "corpus.jsonl"
    idx.index_stream(corp_jsonl)
    idx.compute_idf()

    # Search
    run_path = Path(f"/tmp/run_{args.split}_bm25.txt")
    with open(run_path, "w", encoding="utf-8") as out:
        for qid in sorted(suspicious.keys()):
            results = idx.search(suspicious[qid], top_k=10)
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} bm25\n")

    # Evaluate retrieval
    import math
    r1 = r10 = 0; ndcg10 = 0; mrr_sum = 0
    top10 = defaultdict(list)
    with open(run_path) as f:
        for line in f:
            qid, _, doc, rank = line.strip().split()[:4]
            if int(rank) <= 10:
                top10[qid].append(doc)

    for qid in qrels:
        rel = qrels[qid]
        ranked = top10.get(qid, [])[:10]
        if not ranked:
            continue
        if ranked[0] == rel:
            r1 += 1
        if rel in ranked:
            r10 += 1
        rels = [1 if d == rel else 0 for d in ranked]
        dcg = sum((2**r-1)/math.log2(i+2) for i, r in enumerate(rels))
        ndcg10 += dcg
        for i, d in enumerate(ranked, 1):
            if d == rel:
                mrr_sum += 1.0 / i
                break

    n = len(qrels)
    print(f"  R@10:     {r10/n:.4f}")
    print(f"  nDCG@10:  {ndcg10/n:.4f}")
    print(f"  MRR:      {mrr_sum/n:.4f}")

    # ---- Phase 2: Text Alignment ----
    print("\n--- Phase 2: Text Alignment ---")

    aligner = AlignerV2(
        threshold=args.threshold, jaccard_weight=0.0,
        min_block_len=100, merge_gap=1,
    )

    all_preds = {}
    t0 = time.time()
    for qi, qid in enumerate(sorted(suspicious.keys())):
        susp_text = suspicious[qid]
        # Top-1 source doc
        srcs = [(top10[qid][0], corpus.get(top10[qid][0], ""))] if top10.get(qid) else []
        if not srcs:
            continue
        dets = aligner.align_pair(susp_text, srcs)
        all_preds[qid] = dets
        if (qi + 1) % 20 == 0:
            print(f"  {qi+1}/{len(suspicious)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    total_dets = sum(len(v) for v in all_preds.values())
    print(f"  Done: {len(suspicious)} queries in {elapsed:.0f}s, {total_dets} detections")

    # Evaluate alignment
    if truth_dir.exists():
        truth = build_truth_from_xml(truth_dir)
        total_pred = 0; total_truth = 0; total_hits = 0

        for qid, preds in all_preds.items():
            susp_file = id2file.get(qid, "")
            if susp_file not in truth:
                continue
            truth_list = truth[susp_file]
            total_pred += len(preds)
            total_truth += len(truth_list)

            for t_src, t_ss, t_se, t_sos, t_soe, _ in truth_list:
                for p_ss, p_se, p_src, p_sos, p_soe, _ in preds:
                    o_s = max(t_ss, p_ss); o_e = min(t_se, p_se)
                    if o_e > o_s:
                        o = o_e - o_s
                        u = (t_se - t_ss) + (p_se - p_ss) - o
                        if u > 0 and o / u > 0.3:
                            total_hits += 1
                            break

        prec = total_hits / max(total_pred, 1)
        rec = total_hits / max(total_truth, 1)
        f1 = 2 * prec * rec / max(prec + rec, 0.001)
        print(f"\n  Alignment: Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
        print(f"  Pred={total_pred} Truth={total_truth} Hits={total_hits}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
