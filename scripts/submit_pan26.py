#!/usr/bin/env python3
"""
PAN26 Task 4 submission entrypoint — source retrieval only.

Reads $inputDataset/{corpus.jsonl.gz,queries.jsonl}, writes $outputDir/run.txt.

Pipeline:
  1. BM25 inverted index + retrieval (always, numpy-only, fast)
  2. Query decomposition: paragraph-level BM25 + reciprocal rank fusion
  3. Full chunking + E5 dense retrieval (if corpus < 5000 docs OR GPU available)
  4. RRF merge → TREC run.txt

Usage:
  python submit_pan26.py --input $inputDataset --output $outputDir
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── BM25 (self-contained, no external deps) ──────────────────────────

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = {}
        self.doc_lengths = None
        self.doc_norm = None
        self.doc_id_strs = []
        self.idf = {}
        self.total_docs = 0
        self.avg_dl = 0.0

    def index(self, corpus_path, max_df_ratio=0.3):
        term_df = defaultdict(int)
        doc_ids, doc_lengths = [], []
        opener = gzip.open if str(corpus_path).endswith(".gz") else open
        with opener(corpus_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                did = doc.get("doc_id") or doc.get("qid")
                text = doc.get("default_text") or ""
                doc_ids.append(did)
                tokens = tokenize(text)
                doc_lengths.append(len(tokens))
                for term in set(tokens):
                    term_df[term] += 1
        self.total_docs = len(doc_ids)
        self.doc_id_strs = doc_ids
        self.doc_lengths = np.array(doc_lengths, dtype=np.int32)
        self.avg_dl = float(np.mean(self.doc_lengths))

        max_df = int(self.total_docs * max_df_ratio)
        keep_terms = {}
        total_postings = 0
        for term, df in term_df.items():
            if df > max_df:
                continue
            self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            keep_terms[term] = df
            total_postings += df
        del term_df

        self.doc_norm = (1.0 - self.b + self.b * self.doc_lengths / self.avg_dl).astype(np.float32)
        self._all_indices = np.empty(total_postings, dtype=np.int32)
        self._all_tfs = np.empty(total_postings, dtype=np.float32)
        self._term_slices = {}

        offset = 0
        for term in sorted(keep_terms):
            df = keep_terms[term]
            self._term_slices[term] = (offset, offset + df)
            offset += df
        term_list = sorted(keep_terms)
        term_to_idx = {t: i for i, t in enumerate(term_list)}
        write_pos = np.array([self._term_slices[t][0] for t in term_list], dtype=np.int32)

        with opener(corpus_path, "rt", encoding="utf-8") as f:
            for doc_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("default_text") or ""
                tf_counts = defaultdict(int)
                for t in tokenize(text):
                    if t in keep_terms:
                        tf_counts[t] += 1
                for term, tf in tf_counts.items():
                    tidx = term_to_idx[term]
                    pos = write_pos[tidx]
                    self._all_indices[pos] = doc_idx
                    self._all_tfs[pos] = float(tf)
                    write_pos[tidx] = pos + 1

        del keep_terms, term_to_idx, term_list

    def search(self, query_text, top_k=10):
        tokens = tokenize(query_text)
        if not tokens:
            return []
        scores = np.zeros(self.total_docs, dtype=np.float64)
        for term in tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            sl = self._term_slices.get(term)
            if sl is None:
                continue
            start, end = sl
            indices, tfs = self._all_indices[start:end], self._all_tfs[start:end]
            norm = self.doc_norm[indices]
            tf_norm = tfs * (self.k1 + 1.0) / (tfs + self.k1 * norm)
            scores[indices] += idf * tf_norm
        if top_k >= self.total_docs:
            top = np.argsort(scores)[::-1][:top_k]
        else:
            part = np.argpartition(scores, -top_k)[-top_k:]
            top = part[np.argsort(scores[part])[::-1]]
        return [(self.doc_id_strs[i], float(scores[i])) for i in top if scores[i] > 0]


# ── Query decomposition ──────────────────────────────────────────────

def split_query_sections(query_text):
    """Split query into sections by Markdown headings or paragraph breaks."""
    # Split on ##, ### headings or double-newline paragraph breaks
    sections = re.split(r"\n\n+|\n#{1,3}\s", query_text)
    return [s.strip() for s in sections if len(s.strip()) > 100]


def decompose_query(bm25, query_text, top_k=100):
    """Independent BM25 per section, merge by max reciprocal rank."""
    sections = split_query_sections(query_text)
    if len(sections) <= 1:
        return bm25.search(query_text, top_k=top_k)

    all_candidates = {}
    for sec in sections:
        results = bm25.search(sec, top_k=top_k)
        for rank, (doc_id, score) in enumerate(results, 1):
            rrf = 1.0 / (60 + rank)
            all_candidates[doc_id] = max(all_candidates.get(doc_id, 0), rrf)

    sorted_cands = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, 1.0 / (60 + rank) * 1000) for rank, (doc_id, _) in enumerate(sorted_cands, 1)][:top_k]


# ── Reciprocal Rank Fusion ───────────────────────────────────────────

def rrf_merge(run1, run2, k=60, top_k=10):
    """Merge two ranked lists by RRF, score = 1/(k+rank1) + 1/(k+rank2)."""
    merged = defaultdict(float)
    for rank, (doc_id, _) in enumerate(run1, 1):
        merged[doc_id] += 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(run2, 1):
        merged[doc_id] += 1.0 / (k + rank)
    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in sorted_items[:top_k]]


# ── Chunking + E5 dense retrieval ────────────────────────────────────

CHUNK_SIZE = 256
CHUNK_OVERLAP = 128


def chunk_text(text, doc_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Tokenize and chunk text into overlapping windows."""
    tokens = tokenize(text)
    chunks = []
    for start in range(0, max(1, len(tokens)), chunk_size - overlap):
        chunk_tokens = tokens[start:start + chunk_size]
        if len(chunk_tokens) < 20:
            break
        chunks.append({
            "chunk_id": f"{doc_id}__c{start}",
            "doc_id": doc_id,
            "text": " ".join(chunk_tokens),
        })
    return chunks


def dense_retrieve_with_chunks(model, corpus_path, queries, top_k=100):
    """Chunk all docs, encode, retrieve via cosine similarity."""
    # Load and chunk corpus
    all_chunks = []
    doc_chunk_map = defaultdict(list)
    opener = gzip.open if str(corpus_path).endswith(".gz") else open
    with opener(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            did = doc.get("doc_id") or doc.get("qid")
            text = doc.get("default_text") or ""
            chunks = chunk_text(text, did)
            for c in chunks:
                all_chunks.append(c["text"])
                doc_chunk_map[did].append(len(all_chunks) - 1)

    print(f"  {len(all_chunks)} chunks from {len(doc_chunk_map)} docs", file=sys.stderr)

    # Encode
    chunk_embs = model.encode(all_chunks, show_progress_bar=True, batch_size=64,
                              convert_to_numpy=True, normalize_embeddings=True)

    query_texts = [q["text"] for q in queries]
    query_embs = model.encode(query_texts, show_progress_bar=True, batch_size=16,
                              convert_to_numpy=True, normalize_embeddings=True)

    # Retrieve: max chunk score per doc
    all_results = []
    for qi, q_emb in enumerate(query_embs):
        sims = chunk_embs @ q_emb  # cosine (already normalized)
        doc_scores = defaultdict(float)
        for did, chunk_idxs in doc_chunk_map.items():
            if chunk_idxs:
                doc_scores[did] = float(np.max(sims[chunk_idxs]))
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        all_results.append([(did, sc) for did, sc in sorted_docs[:top_k]])

    return all_results


# ── Main entrypoint ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PAN26 source retrieval submission")
    parser.add_argument("--input", type=Path, required=True,
                        help="$inputDataset directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="$outputDir directory")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results per query in run.txt")
    parser.add_argument("--no-dense", action="store_true",
                        help="Skip dense/chunking, BM25+Decomp only")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Locate corpus (try .gz first, then plain .jsonl)
    corpus_gz = args.input / "corpus.jsonl.gz"
    corpus_plain = args.input / "corpus.jsonl"
    if corpus_gz.exists():
        corpus_path = corpus_gz
    elif corpus_plain.exists():
        corpus_path = corpus_plain
    else:
        print(f"ERROR: corpus.jsonl[.gz] not found in {args.input}", file=sys.stderr)
        sys.exit(1)

    queries_path = args.input / "queries.jsonl"
    if not queries_path.exists():
        print(f"ERROR: queries.jsonl not found in {args.input}", file=sys.stderr)
        sys.exit(1)

    t_total = time.time()

    # ── Load queries ──
    t0 = time.time()
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q.get("qid") or q.get("query_id")
            text = q.get("query") or q.get("default_text") or ""
            queries.append({"qid": qid, "text": text})
    print(f"Loaded {len(queries)} queries ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ── BM25 index + retrieval ──
    t0 = time.time()
    print("Building BM25 index...", file=sys.stderr)
    bm25 = BM25()
    bm25.index(str(corpus_path))
    print(f"  {bm25.total_docs} docs indexed ({time.time()-t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    print("BM25 retrieval + query decomposition...", file=sys.stderr)
    bm25_results = {}
    decomp_results = {}
    for qi, q in enumerate(queries):
        bm25_results[qi] = bm25.search(q["text"], top_k=100)
        decomp_results[qi] = decompose_query(bm25, q["text"], top_k=100)
        if (qi + 1) % 5000 == 0:
            print(f"  {qi+1}/{len(queries)}", file=sys.stderr)
    print(f"  Done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Merge BM25 + Decomp
    merged_results = {}
    for qi in range(len(queries)):
        merged_results[qi] = rrf_merge(bm25_results[qi], decomp_results[qi], k=60, top_k=100)

    # ── Dense chunking (if available + corpus is manageable) ──
    dense_results = None
    if not args.no_dense:
        try:
            print("Loading E5 model...", file=sys.stderr)
            model = SentenceTransformer("intfloat/e5-base-v2")
            import torch
            has_gpu = torch.cuda.is_available()
            device = "cuda" if has_gpu else "cpu"
            model.to(device)
            print(f"  Device: {device}, corpus: {bm25.total_docs} docs", file=sys.stderr)
            dense_raw = dense_retrieve_with_chunks(model, str(corpus_path), queries, top_k=100)
            dense_results = {qi: res for qi, res in enumerate(dense_raw)}
            print("  Dense retrieval done", file=sys.stderr)
        except ImportError:
            print("  sentence-transformers not available, dense skipped", file=sys.stderr)
        except Exception as e:
            print(f"  Dense error: {e}, falling back to BM25-only", file=sys.stderr)

    # ── Final merge ──
    if dense_results:
        final = {}
        for qi in range(len(queries)):
            final[qi] = rrf_merge(merged_results[qi], dense_results[qi], k=60, top_k=args.top_k)
    else:
        final = {qi: merged_results[qi][:args.top_k] for qi in range(len(queries))}

    # ── Write run.txt ──
    run_path = args.output / "run.txt"
    with open(run_path, "w", encoding="utf-8") as out:
        for qi, q in enumerate(queries):
            results = final[qi]
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{q['qid']} Q0 {doc_id} {rank} {score:.6f} pan26-submit\n")

    print(f"Wrote {run_path} ({len(queries)} queries, {args.top_k} results each)", file=sys.stderr)
    print(f"Total time: {time.time()-t_total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    # Try import here so it can fail gracefully
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
    main()
