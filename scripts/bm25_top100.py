"""
Generate BM25 top-100 candidates for cross-encoder reranking.

Two-pass:
  1. Build index + search → get top-100 doc IDs
  2. Read corpus again → extract text snippets for needed docs

Output JSONL: {qid, query_text, candidates: [{doc_id, bm25_score, text}]}
"""

import argparse
import json
import math
import re
import time
import numpy as np
from collections import defaultdict
from pathlib import Path


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class InvertedIndexBM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_lens = []
        self.postings = defaultdict(list)
        self.N = 0
        self.avgdl = 0.0
        self.idf = {}

    def index_stream(self, path):
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("default_text") or doc.get("query") or ""
                tokens = tokenize(text)
                self.doc_ids.append(doc.get("doc_id") or doc.get("qid"))
                self.doc_lens.append(len(tokens))
                self.N += 1

                tf = defaultdict(int)
                for t in tokens:
                    tf[t] += 1
                doc_idx = self.N - 1
                for term, freq in tf.items():
                    self.postings[term].append((doc_idx, freq))

                if self.N % 10000 == 0:
                    print(f"  indexed {self.N} docs ({time.time() - t0:.1f}s)")

        self.avgdl = np.mean(self.doc_lens)
        print(f"  {self.N} docs, {len(self.postings)} terms, avg dl={self.avgdl:.1f}")

    def compute_idf(self):
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query_text, top_k=100, max_query_terms=100):
        qtf = defaultdict(int)
        for t in tokenize(query_text):
            qtf[t] += 1

        term_idfs = [(t, self.idf.get(t, 0)) for t in qtf]
        term_idfs.sort(key=lambda x: x[1], reverse=True)
        top_terms = term_idfs[:max_query_terms]

        doc_scores = defaultdict(float)
        for term, idf in top_terms:
            if idf == 0:
                continue
            for doc_idx, tf in self.postings.get(term, []):
                doc_len = self.doc_lens[doc_idx]
                score = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
                doc_scores[doc_idx] += score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in sorted_docs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--snippet-chars", type=int, default=1500)
    args = parser.parse_args()
    sc = args.snippet_chars

    # ---- Pass 1 ----
    idx = InvertedIndexBM25()
    print("=== Pass 1: Index + Search ===")
    idx.index_stream(args.corpus)
    idx.compute_idf()
    print(f"  vocab: {len(idx.idf)} terms")

    qids, qtexts = [], []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")

    print(f"Searching {len(qids)} queries (top-{args.top_k})...")
    t0 = time.time()
    all_results = []
    needed_docs = set()
    for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
        results = idx.search(qtext, top_k=args.top_k)
        all_results.append((qid, qtext, results))
        for doc_id, _ in results:
            needed_docs.add(doc_id)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(qids)} ({time.time() - t0:.1f}s)")

    print(f"  Done: {len(qids)} queries, {len(needed_docs)} unique docs needed")

    # Free index
    del idx

    # ---- Pass 2: text extraction ----
    print("\n=== Pass 2: Text extraction ===")
    t0 = time.time()
    doc_texts = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("qid")
            if doc_id in needed_docs:
                text = doc.get("default_text") or doc.get("query") or ""
                doc_texts[doc_id] = text[:sc]
                if len(doc_texts) == len(needed_docs):
                    break
    print(f"  {len(doc_texts)} snippets ({time.time() - t0:.1f}s)")

    # ---- Write ----
    print("\n=== Writing ===")
    with open(args.output, "w", encoding="utf-8") as out:
        for qid, qtext, results in all_results:
            entry = {
                "qid": qid,
                "query_text": qtext[:sc * 2],
                "candidates": [
                    {"doc_id": d, "bm25_score": round(s, 4), "text": doc_texts.get(d, "")}
                    for d, s in results
                ],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
