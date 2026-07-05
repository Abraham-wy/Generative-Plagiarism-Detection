"""
Efficient BM25 using inverted index. Streams corpus, builds posting lists,
only scores docs that share terms with the query.

Usage:
  python scripts/bm25_index.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries_5k.jsonl \
    --output data/run_train_5k.txt
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
        self.postings = defaultdict(list)  # term -> [(doc_idx, tf), ...]
        self.N = 0
        self.avgdl = 0.0
        self.idf = {}

    def index_stream(self, path):
        """Single-pass indexing from JSONL file."""
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get("doc_id") or doc.get("qid")
                text = doc.get("default_text") or doc.get("query") or ""
                tokens = tokenize(text)
                self.doc_ids.append(doc_id)
                self.doc_lens.append(len(tokens))
                self.N += 1

                # Count term frequencies in this doc
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

    def search(self, query_text, top_k=10, max_query_terms=100):
        """Score and return top-k results. Uses top max_query_terms by IDF for speed."""
        query_tokens = tokenize(query_text)
        # Count query term frequencies and get unique terms
        qtf = defaultdict(int)
        for t in query_tokens:
            qtf[t] += 1

        # Get terms with their IDF values
        term_idfs = [(t, self.idf.get(t, 0)) for t in qtf]
        term_idfs.sort(key=lambda x: x[1], reverse=True)

        # Use only top max_query_terms
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

        # Sort and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in sorted_docs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    idx = InvertedIndexBM25(k1=args.k1, b=args.b)

    print("Indexing corpus...")
    idx.index_stream(args.corpus)
    print("Computing IDF...")
    idx.compute_idf()
    print(f"  vocab: {len(idx.idf)} terms")

    # Read queries
    print(f"Loading queries from {args.queries}...")
    qids = []
    qtexts = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")

    print(f"  {len(qids)} queries")

    print(f"Retrieving top-{args.top_k}...")
    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
            results = idx.search(qtext, top_k=args.top_k, max_query_terms=100)
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} bm25\n")
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(qids)} ({elapsed:.1f}s, {(i+1)/elapsed:.0f} q/s)")

    elapsed = time.time() - t0
    print(f"Done: {len(qids)} queries in {elapsed:.1f}s")
    print(f"Run: {args.output}")


if __name__ == "__main__":
    main()
