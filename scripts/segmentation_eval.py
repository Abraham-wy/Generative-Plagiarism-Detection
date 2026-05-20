"""
Compare standard BM25 vs query-segmented BM25 (source voting).
Models: query chunk → independent retrieval → source accumulation.

Usage:
  python scripts/segmentation_eval.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries_800_new.jsonl \
    --qrels data/pan25_retrieval/train/qrels.txt
"""
import argparse, json, math, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b
        self.doc_ids = []; self.doc_lens = []
        self.postings = defaultdict(list)
        self.N = 0; self.avgdl = 0.0; self.idf = {}

    def index(self, path):
        t0 = time.time()
        with open(path, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line.strip())
                did = d.get("doc_id") or d.get("qid")
                tokens = tokenize(d.get("default_text") or "")
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N += 1
                tf = defaultdict(int)
                for t in tokens: tf[t] += 1
                for term, freq in tf.items(): self.postings[term].append((self.N - 1, freq))
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1.0
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        print(f"  Indexed {self.N} docs ({time.time() - t0:.0f}s)")

    def search(self, text, top_k=50, max_terms=50):
        tokens = tokenize(text)
        qtf = defaultdict(int)
        for t in tokens: qtf[t] += 1
        terms = sorted([(t, self.idf.get(t, 0)) for t in qtf], key=lambda x: x[1], reverse=True)[:max_terms]
        scores = defaultdict(float)
        for term, idf_val in terms:
            if idf_val == 0: continue
            for doc_idx, tf in self.postings.get(term, []):
                dl = self.doc_lens[doc_idx]
                scores[doc_idx] += idf_val * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in ranked if score > 0]


def paragraph_split(text, min_chars=800, max_chars=3000):
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []; buf = ''
    for p in paras:
        if len(buf) + len(p) > max_chars and len(buf) > min_chars:
            chunks.append(buf.strip()); buf = p
        else:
            buf = buf + '\n\n' + p if buf else p
    if buf.strip() and len(buf.strip()) > min_chars: chunks.append(buf.strip())
    return chunks if chunks else [text[:max_chars]]


def semantic_split(text):
    sections = re.split(
        r'\n\s*(?:\d+\.?\s*)?(?:A[Bb][Ss][Tt][Rr][Aa][Cc][Tt]|'
        r'[Ii][Nn][Tt][Rr][Oo][Dd][Uu][Cc][Tt][Ii][Oo][Nn]|'
        r'[Rr][Ee][Ll][Aa][Tt][Ee][Dd]\s*[Ww][Oo][Rr][Kk]|'
        r'[Mm][Ee][Tt][Hh][Oo][Dd]|'
        r'[Ee][Xx][Pp][Ee][Rr][Ii][Mm][Ee][Nn][Tt]|'
        r'[Rr][Ee][Ss][Uu][Ll][Tt]|'
        r'[Dd][Ii][Ss][Cc][Uu][Ss][Ss][Ii][Oo][Nn]|'
        r'[Cc][Oo][Nn][Cc][Ll][Uu][Ss][Ii][Oo][Nn]|'
        r'[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee]|'
        r'[Aa][Cc][Kk][Nn][Oo][Ww][Ll][Ee][Dd][Gg])\s*\n',
        text, flags=re.IGNORECASE)
    chunks = [s.strip() for s in sections if len(s.strip()) > 500]
    return chunks[:15] if chunks else paragraph_split(text)


def vote_aggregate(all_results):
    """Coverage-weighted voting: score = sum(scores) * (1 + n_matched/n_total)"""
    n_total = len(all_results)
    merged_scores = defaultdict(float)
    merged_count = defaultdict(int)
    for results in all_results:
        seen = set()
        for doc_id, score in results:
            merged_scores[doc_id] += score
            if doc_id not in seen:
                merged_count[doc_id] += 1
                seen.add(doc_id)
    final = {}
    for doc_id in merged_scores:
        coverage = merged_count[doc_id] / n_total
        final[doc_id] = merged_scores[doc_id] * (1.0 + coverage)
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


def compute_metrics(qrels, run, k=10):
    """Compute R@K, nDCG@K, MRR"""
    n = len(qrels)
    r = nd = mr = 0.0
    for qid, rel_doc in qrels.items():
        ranked = run.get(qid, [])
        if not ranked: continue
        if rel_doc in ranked[:k]: r += 1
        rels = [1 if d == rel_doc else 0 for d in ranked[:k]]
        nd += sum((2 ** r_ - 1) / math.log2(i + 2) for i, r_ in enumerate(rels))
        for i, d in enumerate(ranked[:k], 1):
            if d == rel_doc:
                mr += 1.0 / i
                break
    return {
        f'R@{k}': r / n,
        f'nDCG@{k}': nd / n,
        'MRR': mr / n,
    }


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--strategies", default="paragraph_split,semantic_split")
    args = parser.parse_args()
    print_ = lambda *a, **kw: (print(*a, **kw, flush=True), sys.stdout.flush())

    # Load qrels
    qrels = {}
    with open(args.qrels) as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0: qrels[qid] = doc

    # Load queries
    qids, qtexts = [], []
    with open(args.queries, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            if qid in qrels:
                qids.append(qid)
                qtexts.append(q.get("query") or q.get("default_text") or "")
    print(f"Loaded {len(qids)} queries with qrels")

    # Build index
    print("Building BM25 index...")
    idx = BM25Index()
    idx.index(args.corpus)

    # ---- Standard BM25 ----
    print("\nRunning standard BM25...")
    t0 = time.time()
    run_std = {}
    segmented_count = 0
    for qi, (qid, qtext) in enumerate(zip(qids, qtexts)):
        chunks = paragraph_split(qtext)
        if len(chunks) > 1:
            segmented_count += 1
            all_results = [idx.search(c, top_k=50) for c in chunks]
            results = vote_aggregate(all_results)[:100]
        else:
            results = idx.search(qtext, top_k=100)
        run_std[qid] = [d for d, _ in results]
        if (qi + 1) % 400 == 0:
            print(f"  {qi + 1}/{len(qids)} ({time.time() - t0:.0f}s)")
    t_std = time.time() - t0

    metrics_std = compute_metrics(qrels, run_std, k=10)
    metrics_std100 = compute_metrics(qrels, run_std, k=100)

    # ---- True Standard BM25 (no segmentation) ----
    print("\nRunning standard BM25 (no segmentation)...")
    t0 = time.time()
    run_base = {}
    for qi, (qid, qtext) in enumerate(zip(qids, qtexts)):
        results = idx.search(qtext, top_k=100)
        run_base[qid] = [d for d, _ in results]
        if (qi + 1) % 400 == 0:
            print(f"  {qi + 1}/{len(qids)} ({time.time() - t0:.0f}s)")
    t_base = time.time() - t0

    metrics_base = compute_metrics(qrels, run_base, k=10)
    metrics_base100 = compute_metrics(qrels, run_base, k=100)

    # ---- Semantic split ----
    print("\nRunning semantic split...")
    t0 = time.time()
    run_sem = {}
    sem_segmented = 0
    for qi, (qid, qtext) in enumerate(zip(qids, qtexts)):
        chunks = semantic_split(qtext)
        if len(chunks) > 1:
            sem_segmented += 1
            all_results = [idx.search(c, top_k=50) for c in chunks]
            results = vote_aggregate(all_results)[:100]
        else:
            results = idx.search(qtext, top_k=100)
        run_sem[qid] = [d for d, _ in results]
        if (qi + 1) % 400 == 0:
            print(f"  {qi + 1}/{len(qids)} ({time.time() - t0:.0f}s)")
    t_sem = time.time() - t0

    metrics_sem = compute_metrics(qrels, run_sem, k=10)
    metrics_sem100 = compute_metrics(qrels, run_sem, k=100)

    # ---- Report ----
    nq = len(qids)
    pct_para = 100 * segmented_count / nq if nq else 0
    pct_sem = 100 * sem_segmented / nq if nq else 0

    print(f"\n{'='*60}")
    print(f"Results on {nq} queries")
    print(f"  paragraph_split: {segmented_count}/{nq} ({pct_para:.1f}%) queries segmented")
    print(f"  semantic_split:   {sem_segmented}/{nq} ({pct_sem:.1f}%) queries segmented")
    print(f"  Time: std={t_base:.0f}s, para={t_std:.0f}s, sem={t_sem:.0f}s")
    print(f"\n{'Metric':<15} {'Std BM25':>10} {'ParaSeg':>10} {'SemSeg':>10}")
    print(f"{'-'*45}")
    for m in ['R@10', 'nDCG@10', 'MRR', 'R@100']:
        b = metrics_base.get(m, 0)
        p = metrics_std.get(m, 0)
        s = metrics_sem.get(m, 0)
        print(f"{m:<15} {b:>10.4f} {p:>10.4f} {s:>10.4f}")

    print(f"\n{'Diff vs Std':<15} {'ParaSeg':>10} {'SemSeg':>10}")
    print(f"{'-'*35}")
    for m in ['R@10', 'nDCG@10', 'MRR', 'R@100']:
        b = metrics_base.get(m, 0)
        p = metrics_std.get(m, 0) - b
        s = metrics_sem.get(m, 0) - b
        print(f"{m:<15} {p:>+10.4f} {s:>+10.4f}")


if __name__ == "__main__":
    main()
