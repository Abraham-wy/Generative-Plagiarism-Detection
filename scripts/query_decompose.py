"""
Cat 1 query decomposition: split long queries into sections, independent BM25,
merge by max reciprocal rank.

For queries where source doc is NOT in BM25 top-100, decomposing the
54K-char query into focused sections helps each section find the right doc.

Usage:
  python scripts/query_decompose.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --qrels data/pan25_retrieval/train/qrels.txt \
    --bm25-run data/run_train_bm25.txt \
    --output data/run_train_decomposed.txt
"""

import argparse, json, math, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25Quick:
    """Lightweight BM25 for section-level retrieval."""
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b
        self.doc_ids = []; self.doc_lens = []
        self.postings = defaultdict(list)
        self.N = 0; self.avgdl = 0.0; self.idf = {}

    def index(self, corpus_path):
        t0 = time.time()
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                did = d.get("doc_id") or d.get("qid")
                text = d.get("default_text") or ""
                tokens = tokenize(text)
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N += 1
                tf = defaultdict(int)
                for t in tokens: tf[t] += 1
                for term, freq in tf.items():
                    self.postings[term].append((self.N - 1, freq))
                if self.N % 10000 == 0:
                    print(f"  indexed {self.N} docs ({time.time()-t0:.1f}s)")
        self.avgdl = np.mean(self.doc_lens)
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        print(f"  {self.N} docs, {len(self.idf)} terms, avg_dl={self.avgdl:.0f}")

    def search(self, query_text, top_k=100):
        qt = tokenize(query_text)
        qtf = defaultdict(int)
        for t in qt: qtf[t] += 1
        term_idfs = sorted([(t, self.idf.get(t, 0)) for t in qtf], key=lambda x: x[1], reverse=True)[:100]
        scores = defaultdict(float)
        for term, idf in term_idfs:
            if idf == 0: continue
            for doc_idx, tf in self.postings.get(term, []):
                dl = self.doc_lens[doc_idx]
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], rank + 1, score) for rank, (idx, score) in enumerate(ranked)]


def split_sections(text, min_chars=500):
    """Split text into sections by paragraph breaks, min 500 chars each."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    sections = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) > 3000 and len(buf) > min_chars:
            sections.append(buf.strip())
            buf = p
        else:
            buf = buf + "\n\n" + p if buf else p
    if buf.strip() and len(buf.strip()) > min_chars:
        sections.append(buf.strip())
    if not sections:
        sections = [text[:3000]]
    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--bm25-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load qrels, find Cat 1
    print("Finding Cat 1 queries...")
    qrels = {}
    with open(args.qrels) as f:
        for line in f:
            qid, _, doc, rel = line.strip().split()
            if int(rel) > 0: qrels[qid] = doc

    bm25_100 = defaultdict(set)
    with open("data/bm25_top100_train.jsonl") as f:
        for line in f:
            q = json.loads(line.strip())
            for c in q["candidates"]:
                bm25_100[q["qid"]].add(c["doc_id"])

    cat1 = [qid for qid in qrels if qrels[qid] not in bm25_100.get(qid, set())]
    print(f"  Cat 1 (src not in BM25@100): {len(cat1)}")

    # Index corpus (shared across all queries)
    print("\nIndexing corpus with BM25...")
    bm25 = BM25Quick()
    bm25.index(args.corpus)

    # Load Cat 1 queries
    cat1_qtexts = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            if qid in cat1:
                cat1_qtexts[qid] = q.get("query") or q.get("default_text") or ""

    # Load full BM25 run (to keep non-Cat-1 queries intact)
    bm25_run = {}
    with open(args.bm25_run) as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            if qid not in bm25_run:
                bm25_run[qid] = []
            if len(bm25_run[qid]) < 10:
                bm25_run[qid].append((parts[2], float(parts[4])))

    # ---- Decompose and search ----
    print(f"\nDecomposing {len(cat1_qtexts)} Cat 1 queries...")
    t0 = time.time()
    decomposed_results = {}

    for qi, (qid, qtext) in enumerate(sorted(cat1_qtexts.items())):
        sections = split_sections(qtext)
        if len(sections) <= 1:
            # Can't decompose meaningfully; keep original BM25 results
            decomposed_results[qid] = bm25_run.get(qid, [])
            continue

        # Search each section independently
        merged = {}
        for sec_idx, section in enumerate(sections):
            results = bm25.search(section, top_k=50)
            for doc_id, rank, score in results:
                rr = 1.0 / (60 + rank)  # reciprocal rank
                if doc_id not in merged:
                    merged[doc_id] = 0.0
                merged[doc_id] = max(merged[doc_id], rr)  # max RR across sections

        top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]
        decomposed_results[qid] = [(doc_id, rr) for doc_id, rr in top]

        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(cat1_qtexts)} ({time.time()-t0:.1f}s)")

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ---- Write output ----
    print(f"Writing {args.output}...")
    with open(args.output, "w", encoding="utf-8") as out:
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                qid = q.get("qid") or q.get("query_id")
                if qid in decomposed_results:
                    for rank, (did, score) in enumerate(decomposed_results[qid], 1):
                        out.write(f"{qid} Q0 {did} {rank} {score:.6f} decompose\n")
                elif qid in bm25_run:
                    for rank, (did, score) in enumerate(bm25_run[qid], 1):
                        out.write(f"{qid} Q0 {did} {rank} {score:.6f} bm25\n")

    # Eval
    cat1_hits = 0
    for qid in cat1:
        if qid in decomposed_results:
            if any(d == qrels[qid] for d, _ in decomposed_results[qid][:10]):
                cat1_hits += 1
    print(f"\nCat 1 recovered @10: {cat1_hits}/{len(cat1)} ({100*cat1_hits/max(len(cat1),1):.1f}%)")

    bm25_hits = sum(1 for qid in qrels if qrels[qid] in {d for d,_ in bm25_run.get(qid,[])[:10]})
    all_hits = bm25_hits + cat1_hits
    print(f"BM25@10: {bm25_hits}/{len(qrels)} = {100*bm25_hits/len(qrels):.1f}%")
    print(f"+Decompose@10: {all_hits}/{len(qrels)} = {100*all_hits/len(qrels):.1f}%")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
