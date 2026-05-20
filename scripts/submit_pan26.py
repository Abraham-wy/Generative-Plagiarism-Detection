#!/usr/bin/env python3
"""
PAN26 Task 4 submission: query-segmented BM25 with source voting.

Query-chunk → independent retrieval → source accumulation = provenance graph.
"""
import argparse, gzip, heapq, json, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b
        self.doc_ids = []; self.doc_lens = []
        self.postings = defaultdict(list)
        self.N = 0; self.avgdl = 0.0; self.idf = {}; self.df = {}

    def index(self, path):
        t0 = time.time()
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                did = d.get("doc_id") or d.get("qid")
                tokens = tokenize(d.get("default_text") or "")
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N += 1
                tf = defaultdict(int)
                for t in tokens: tf[t] += 1
                for term, freq in tf.items(): self.postings[term].append((self.N - 1, freq))
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1.0
        for term, posting in self.postings.items():
            df = len(posting)
            self.df[term] = df
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        print(f"  Indexed {self.N} docs ({time.time() - t0:.0f}s)")

    def search(self, text, top_k=10, max_terms=100, max_df=5000):
        tokens = tokenize(text)
        qtf = defaultdict(int)
        for t in tokens: qtf[t] += 1
        candidates = [(t, self.idf.get(t, 0)) for t in qtf if self.df.get(t, 0) <= max_df]
        terms = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_terms]
        scores = defaultdict(float)
        for term, idf_val in terms:
            if idf_val == 0: continue
            for doc_idx, tf in self.postings.get(term, []):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(os.environ.get("inputDataset", "data/pan26/test-dataset")))
    p.add_argument("--output", type=Path, default=Path(os.environ.get("outputDir", "/tmp/pan26_output")))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--no-segment", action="store_true", help="Disable query segmentation (pure BM25)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    corp = args.input / "corpus.jsonl"
    if not corp.exists(): corp = args.input / "corpus.jsonl.gz"
    queries_path = args.input / "queries.jsonl"

    print(f"Corpus: {corp}")
    print(f"Queries: {queries_path}")

    idx = BM25()
    idx.index(corp)

    qids, qtexts = [], []
    with open(queries_path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line.strip())
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")
    print(f"  {len(qids)} queries")

    out = args.output / "run.txt"
    t0 = time.time()
    segmented = 0

    with open(out, 'w', encoding='utf-8') as f:
        for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
            if args.no_segment:
                results = idx.search(qtext, top_k=args.top_k, max_terms=100)
            else:
                chunks = paragraph_split(qtext)
                if len(chunks) > 1:
                    segmented += 1
                    all_results = [idx.search(c, top_k=50, max_terms=20) for c in chunks[:20]]
                    results = vote_aggregate(all_results)[:args.top_k]
                else:
                    results = idx.search(qtext, top_k=args.top_k, max_terms=100)

            for rank, (doc_id, score) in enumerate(results, 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} seg\n")
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                qps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i + 1}/{len(qids)} ({elapsed:.0f}s, {qps:.1f} q/s)")

    elapsed = time.time() - t0
    print(f"Done: {len(qids)} queries in {elapsed:.0f}s ({segmented} segmented)")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
