"""
BM25 retriever — memory-efficient for large corpora (60K+ docs).

Key design:
  - Two-pass corpus streaming (never loads full corpus into memory)
  - Flat numpy arrays for ALL posting lists (no per-term Python objects)
  - numpy vectorized scoring
  - Long query truncation (--query-max-chars, default 2000)

  Peak memory for 60K docs: ~2GB (flat posting arrays + doc metadata)

Usage:
  python scripts/baseline_bm25.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --output data/runs/bm25_train.txt
"""

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def _stream_corpus(path):
    """Generator yielding (doc_id_str, text)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("qid")
            text = doc.get("default_text") or doc.get("query") or ""
            yield doc_id, text


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

        # Flat posting arrays: all terms' postings concatenated
        self._all_indices = None   # int32 array
        self._all_tfs = None       # float32 array
        # term -> slice(start, end) into flat arrays
        self._term_slices = {}
        # term -> idf (float)
        self.idf = {}
        # doc_idx -> length  (int32 array)
        self.doc_lengths = None
        # doc_idx -> norm: (1 - b + b * len/avgdl)  (float32 array)
        self.doc_norm = None
        # doc_idx -> original string ID
        self.doc_id_strs = []
        self.total_docs = 0
        self.avg_dl = 0.0

    def index(self, corpus_path, max_df_ratio=0.3):
        """Index from corpus.jsonl path. Two-pass streaming."""

        # ============================================================
        # Pass 1: count doc lengths, collect doc IDs, count term DF
        # ============================================================
        print("    Pass 1: counting term frequencies...")
        t0 = time.time()
        term_df = defaultdict(int)
        doc_ids = []
        doc_lengths = []

        for doc_id_str, text in _stream_corpus(corpus_path):
            doc_ids.append(doc_id_str)
            tokens = tokenize(text)
            doc_lengths.append(len(tokens))
            for term in set(tokens):
                term_df[term] += 1

        self.total_docs = len(doc_ids)
        self.doc_id_strs = doc_ids
        self.doc_lengths = np.array(doc_lengths, dtype=np.int32)
        self.avg_dl = float(np.mean(self.doc_lengths))
        n_unique = len(term_df)
        print(f"    {self.total_docs} docs, {n_unique} unique terms "
              f"({time.time() - t0:.1f}s), avg_dl={self.avg_dl:.0f}")

        # ============================================================
        # Prune stop words, compute IDF, pre-allocate flat arrays
        # ============================================================
        max_df = int(self.total_docs * max_df_ratio)
        keep_terms = {}
        dropped = 0
        total_postings = 0
        for term, df in term_df.items():
            if df > max_df:
                dropped += 1
            else:
                self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                keep_terms[term] = df
                total_postings += df
        print(f"    {len(keep_terms)} terms kept, {dropped} pruned, "
              f"{total_postings:,} total postings")

        del term_df

        # Precompute per-doc norm
        if self.avg_dl > 0:
            self.doc_norm = (1.0 - self.b + self.b * self.doc_lengths / self.avg_dl).astype(np.float32)

        # Allocate flat posting arrays
        self._all_indices = np.empty(total_postings, dtype=np.int32)
        self._all_tfs = np.empty(total_postings, dtype=np.float32)

        # Assign each term a contiguous slice in the flat arrays
        offset = 0
        for term in sorted(keep_terms):  # deterministic order
            df = keep_terms[term]
            self._term_slices[term] = (offset, offset + df)
            offset += df

        # Per-term write position counter (int32 array indexed by term order)
        term_list = sorted(keep_terms)
        term_to_idx = {t: i for i, t in enumerate(term_list)}
        write_pos = np.zeros(len(term_list), dtype=np.int32)
        # Pre-populate write_pos with start offsets
        for i, term in enumerate(term_list):
            start, _ = self._term_slices[term]
            write_pos[i] = start
        term_for_idx = term_list  # idx -> term string

        # ============================================================
        # Pass 2: fill flat posting arrays
        # ============================================================
        print("    Pass 2: filling posting arrays...")
        t0 = time.time()

        for doc_idx, (_, text) in enumerate(_stream_corpus(corpus_path)):
            tokens = tokenize(text)

            # Count term frequencies for this doc
            tf_counts = defaultdict(int)
            for t in tokens:
                if t in keep_terms:
                    tf_counts[t] += 1

            for term, tf in tf_counts.items():
                tidx = term_to_idx[term]
                pos = write_pos[tidx]
                self._all_indices[pos] = doc_idx
                self._all_tfs[pos] = float(tf)
                write_pos[tidx] = pos + 1

        del keep_terms, term_to_idx, term_list
        print(f"    Done ({time.time() - t0:.1f}s)")

    def _get_postings(self, term):
        """Return (indices_view, tfs_view) for a term, or None."""
        sl = self._term_slices.get(term)
        if sl is None:
            return None
        start, end = sl
        return self._all_indices[start:end], self._all_tfs[start:end]

    def search(self, query_text, top_k=10):
        """Return [(doc_id_str, score), ...] for top-k docs."""
        tokens = tokenize(query_text)
        if not tokens:
            return []

        scores = np.zeros(self.total_docs, dtype=np.float64)

        for term in tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            postings = self._get_postings(term)
            if postings is None:
                continue
            indices, tfs = postings
            norm = self.doc_norm[indices]
            tf_norm = tfs * (self.k1 + 1.0) / (tfs + self.k1 * norm)
            scores[indices] += idf * tf_norm

        if top_k >= self.total_docs:
            top_indices = np.argsort(scores)[::-1][:top_k]
        else:
            part_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = part_idx[np.argsort(scores[part_idx])[::-1]]

        results = []
        for idx in top_indices:
            s = scores[idx]
            if s > 0:
                results.append((self.doc_id_strs[idx], float(s)))
        return results


def load_queries(path):
    """Generator yielding (qid, text) from queries.jsonl."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q.get("qid") or q.get("query_id")
            text = q.get("query") or q.get("default_text") or ""
            yield qid, text


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline retriever")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-df-ratio", type=float, default=0.3,
                        help="Prune terms appearing in > this fraction of docs (default 0.3)")
    parser.add_argument("--query-max-chars", type=int, default=2000,
                        help="Truncate query to first N chars (0=no truncation, default 2000)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Building BM25 index...")
    bm25 = BM25(k1=args.k1, b=args.b)
    bm25.index(str(args.corpus), max_df_ratio=args.max_df_ratio)
    print(f"Total index time: {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"Loading queries from {args.queries}...")
    queries = list(load_queries(args.queries))
    print(f"  Loaded {len(queries)} queries ({time.time() - t0:.1f}s)")

    qmax = args.query_max_chars
    trunc_note = f", truncated to {qmax} chars" if qmax > 0 else ""
    print(f"Retrieving top-{args.top_k}{trunc_note}...")
    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, (qid, qtext) in enumerate(queries):
            q = qtext[:qmax] if qmax > 0 else qtext
            results = bm25.search(q, top_k=args.top_k)
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} bm25\n")
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(queries)} ({len(queries)/elapsed:.0f} q/s)")

    elapsed = time.time() - t0
    print(f"Done: {len(queries)} queries in {elapsed:.1f}s ({len(queries)/elapsed:.0f} q/s)")
    print(f"Run written to {args.output}")


if __name__ == "__main__":
    main()
