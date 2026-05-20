"""
Compare standard BM25 vs query-segmented (paragraph_split) BM25.
Query-chunk → independent retrieval → source voting = provenance graph.
"""
import argparse, heapq, json, math, re, sys, time
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
        self.N = 0; self.avgdl = 0.0; self.idf = {}; self.df = {}

    def index_stream(self, path):
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                did = d.get("doc_id") or d.get("qid")
                tokens = tokenize(d.get("default_text") or "")
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N += 1
                tf = defaultdict(int)
                for t in tokens: tf[t] += 1
                doc_idx = self.N - 1
                for term, freq in tf.items():
                    self.postings[term].append((doc_idx, freq))
                if self.N % 10000 == 0:
                    print(f"  indexed {self.N} docs ({time.time()-t0:.0f}s)", flush=True)
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1.0
        print(f"  {self.N} docs, {len(self.postings)} terms, avg dl={self.avgdl:.0f}", flush=True)

    def compute_idf(self):
        t0 = time.time()
        for term, posting in self.postings.items():
            df = len(posting)
            self.df[term] = df
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        print(f"  IDF computed: {len(self.idf)} terms ({time.time()-t0:.0f}s)", flush=True)

    def search(self, text, top_k=50, max_terms=50, max_df=5000):
        tokens = tokenize(text)
        qtf = defaultdict(int)
        for t in tokens: qtf[t] += 1
        # Select top terms by IDF, skip terms that appear in too many docs
        candidates = [(t, self.idf.get(t, 0)) for t in qtf if self.df.get(t, 0) <= max_df]
        terms = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_terms]
        scores = defaultdict(float)
        for term, idf_val in terms:
            if idf_val == 0: continue
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_lens[doc_idx]
                scores[doc_idx] += idf_val * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
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


def vote_aggregate(all_results):
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


def compute_metrics(qrels, run):
    n = sum(1 for qid in qrels if qid in run)
    if n == 0: return {'R@10': 0, 'R@100': 0, 'nDCG@10': 0, 'MRR': 0}
    r10 = r100 = nd10 = mr = 0.0
    for qid, rel_doc in qrels.items():
        ranked = run.get(qid, [])
        if not ranked: continue
        if rel_doc in ranked[:10]: r10 += 1
        if rel_doc in ranked[:100]: r100 += 1
        rels = [1 if d == rel_doc else 0 for d in ranked[:10]]
        nd10 += sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
        for i, d in enumerate(ranked[:10], 1):
            if d == rel_doc:
                mr += 1.0 / i
                break
    return {
        'R@10': r10 / n, 'R@100': r100 / n,
        'nDCG@10': nd10 / n, 'MRR': mr / n,
    }


def run_retrieval(idx, qids, qtexts, split_fn, top_k=100):
    """Run retrieval with optional segmentation. Returns {qid: [doc_id, ...]}"""
    run = {}
    segmented = 0
    for qi, (qid, qtext) in enumerate(zip(qids, qtexts)):
        if split_fn:
            chunks = split_fn(qtext)
            if len(chunks) > 1:
                segmented += 1
                all_results = [idx.search(c, top_k=50, max_terms=20) for c in chunks[:20]]
                results = vote_aggregate(all_results)[:top_k]
            else:
                results = idx.search(qtext, top_k=top_k)
        else:
            results = idx.search(qtext, top_k=top_k)
        run[qid] = [d for d, _ in results]
        if (qi + 1) % 200 == 0:
            print(f"    {qi+1}/{len(qids)}", flush=True)
    return run, segmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    # Load qrels
    qrels = {}
    with open(args.qrels) as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0:
                qrels[qid] = doc

    # Load queries (only those with qrels)
    qids, qtexts = [], []
    with open(args.queries, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            if qid in qrels:
                qids.append(qid)
                qtexts.append(q.get("query") or q.get("default_text") or "")

    label = f" [{args.label}]" if args.label else ""
    print(f"Queries with qrels: {len(qids)}{label}", flush=True)

    # Build index ONCE
    print("Building BM25 index...", flush=True)
    t0 = time.time()
    idx = BM25Index()
    idx.index_stream(args.corpus)
    idx.compute_idf()
    print(f"  Index + IDF: {time.time()-t0:.0f}s", flush=True)

    # ---- Standard BM25 (no segmentation) ----
    print("\n[1/2] Standard BM25...", flush=True)
    t0 = time.time()
    run_base, _ = run_retrieval(idx, qids, qtexts, split_fn=None)
    t_base = time.time() - t0
    print(f"  Done: {t_base:.0f}s", flush=True)

    # ---- Paragraph-split segmentation ----
    print("\n[2/2] Paragraph-split segmented BM25...", flush=True)
    t0 = time.time()
    run_seg, n_seg = run_retrieval(idx, qids, qtexts, split_fn=paragraph_split)
    t_seg = time.time() - t0
    print(f"  Done: {t_seg:.0f}s ({n_seg}/{len(qids)} queries segmented)", flush=True)

    # ---- Evaluate ----
    m_base = compute_metrics(qrels, run_base)
    m_seg = compute_metrics(qrels, run_seg)

    print(f"\n{'='*55}")
    print(f"Results on {len(qids)} queries{label}")
    print(f"Segmented queries: {n_seg}/{len(qids)} ({100*n_seg/max(len(qids),1):.1f}%)")
    print(f"{'='*55}")
    print(f"{'Metric':<15} {'Std BM25':>10} {'ParaSeg':>10} {'Delta':>10}")
    print(f"{'-'*45}")
    for m in ['R@10', 'R@100', 'nDCG@10', 'MRR']:
        b = m_base[m]; s = m_seg[m]; d = s - b
        print(f"{m:<15} {b:>10.4f} {s:>10.4f} {d:>+10.4f}", flush=True)
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
