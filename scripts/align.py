"""
Text alignment module for PAN26 Task 4.

Given suspicious doc + top-K source docs (from source retrieval),
detect plagiarized passages with character-level offsets.

Pipeline:
  1. Sentence splitting with char offsets
  2. E5 sentence embeddings + cosine similarity matrix
  3. Threshold detection + adjacent sentence merging (yukino method)
  4. Output PAN-standard alignment XML

Usage:
  python scripts/align.py \
    --suspicious data/pan25_retrieval/spot_check/queries.jsonl \
    --corpus data/pan25_retrieval/spot_check/corpus.jsonl \
    --id-mapping data/pan25_retrieval/spot_check/id_mapping.tsv \
    --run data/run_spot_check.txt \
    --truth data/pan25_retrieval/spot_check/qrels.txt \
    --output data/align_spot_check.xml \
    --pan25-xml-dir pan25/00_spot_check/00_spot_check_truth
"""

import argparse
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from collections import defaultdict


def sent_tokenize(text):
    """Split text into sentences with (start_char, end_char, text) tuples."""
    # Match sentence boundaries: .!? followed by space+capital or newline
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|\n\n+'
    parts = re.split(pattern, text)
    sentences = []
    pos = 0
    for part in parts:
        part = part.strip()
        if not part:
            # Update position tracking for whitespace
            pos = text.find(part, pos) if part else pos
            continue
        start = text.find(part, pos)
        if start == -1:
            start = pos
        end = start + len(part)
        sentences.append((start, end, part))
        pos = end
    return sentences


def detect_paragraphs_by_newlines(text, max_chars_per_para=500):
    """
    Split long text into workable chunks by paragraph boundaries.
    Returns list of (start_char, end_char, text).
    """
    paras = []
    pos = 0
    for para_text in re.split(r'\n\s*\n', text):
        para_text = para_text.strip()
        if not para_text:
            continue
        start = text.find(para_text, pos)
        if start == -1:
            start = pos
        end = start + len(para_text)
        # Split very long paragraphs
        if len(para_text) > max_chars_per_para:
            # Split by sentences within the paragraph
            sub_sents = sent_tokenize(para_text)
            for sub_start, sub_end, sub_text in sub_sents:
                paras.append((start + sub_start, start + sub_end, sub_text))
        else:
            paras.append((start, end, para_text))
        pos = end
    return paras


class Aligner:
    def __init__(self, model_name="intfloat/e5-base-v2", threshold=0.65,
                 min_block_len=50, merge_gap=2):
        self.model_name = model_name
        self.threshold = threshold
        self.min_block_len = min_block_len
        self.merge_gap = merge_gap  # max sentences gap to merge
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = 512
        return self._model

    def encode_sentences(self, sentences):
        """Encode a list of sentence texts. Returns numpy array."""
        if not sentences:
            return np.array([])
        texts = [s[2] for s in sentences]
        emb = self.model.encode(texts, batch_size=64, show_progress_bar=False,
                                 normalize_embeddings=True)
        return emb

    def align_pair(self, susp_text, source_texts):
        """
        Align suspicious doc against multiple source docs.

        Args:
          susp_text: suspicious document full text
          source_texts: list of (source_doc_id, source_full_text) tuples

        Returns:
          detections: list of (susp_start, susp_end, src_doc_id, src_start, src_end, score)
        """
        # Split suspicious into sentences
        susp_sents = sent_tokenize(susp_text)
        if not susp_sents:
            return []

        # Encode suspicious sentences
        susp_emb = self.encode_sentences(susp_sents)

        detections = []

        # Process each source document
        for src_doc_id, src_text in source_texts[:10]:  # top-10 sources
            # Split source into paragraphs (avoid processing full long doc)
            src_chunks = detect_paragraphs_by_newlines(src_text, max_chars_per_para=500)
            if not src_chunks:
                continue

            # Encode source chunks
            src_emb = self.encode_sentences(src_chunks)

            # Similarity matrix
            sim_matrix = susp_emb @ src_emb.T  # (n_susp, n_src)

            # For each suspicious sentence, find best matching source chunk
            best_src_idx = np.argmax(sim_matrix, axis=1)
            best_scores = np.max(sim_matrix, axis=1)

            # Mark sentences above threshold
            marked = best_scores >= self.threshold
            if not marked.any():
                continue

            # Merge adjacent marked sentences with same source chunk
            blocks = self._merge_adjacent(susp_sents, src_chunks, marked,
                                          best_src_idx, best_scores,
                                          src_doc_id)
            detections.extend(blocks)

        # Sort by suspicious position
        detections.sort(key=lambda d: d[0])
        return detections

    def _merge_adjacent(self, susp_sents, src_chunks, marked, best_src_idx, best_scores, src_doc_id):
        """Merge adjacent marked sentences into detection blocks."""
        blocks = []
        i = 0
        while i < len(marked):
            if not marked[i]:
                i += 1
                continue

            # Start a new block
            block_start = i
            block_end = i + 1
            block_scores = [best_scores[i]]
            src_indices = [best_src_idx[i]]

            # Extend forward
            j = i + 1
            while j < len(marked):
                if not marked[j]:
                    j += 1
                    continue
                # Check if close enough to merge
                gap = j - (block_start + len(src_indices))
                if gap > self.merge_gap:
                    break
                # Check same source document (via chunk proximity)
                if abs(best_src_idx[j] - best_src_idx[i]) > 3:  # nearby chunks
                    j += 1
                    continue

                src_indices.append(best_src_idx[j])
                block_scores.append(best_scores[j])
                block_end = j + 1
                j += 1

            # Check minimum length
            char_len = sum(
                susp_sents[k][1] - susp_sents[k][0]
                for k in range(block_start, min(block_end, len(susp_sents)))
            )
            if char_len < self.min_block_len:
                i = block_end
                continue

            # Compute source offsets
            src_start = min(src_chunks[idx][0] for idx in src_indices if idx < len(src_chunks))
            src_end = max(src_chunks[idx][1] for idx in src_indices if idx < len(src_chunks))

            susp_start = susp_sents[block_start][0]
            susp_end = susp_sents[min(block_end - 1, len(susp_sents) - 1)][1]
            avg_score = float(np.mean(block_scores))

            blocks.append((susp_start, susp_end, src_doc_id, src_start, src_end, avg_score))
            i = block_end

        return blocks


def load_id_mapping(path):
    """Load id_mapping.tsv: kind, split, opaque_id, filename"""
    id2file = {}
    file2id = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                opaque_id = parts[2]
                filename = parts[3]
                id2file[opaque_id] = filename
                file2id[filename] = opaque_id
    return id2file, file2id


def build_detections_from_truth(truth_dir, id2file=None):
    """
    Parse PAN25 truth XMLs to get ground-truth detections.
    Returns dict: suspicious_filename -> [(src_filename, susp_start, susp_end, src_start, src_end), ...]
    """
    detections = defaultdict(list)
    for xml_path in sorted(Path(truth_dir).glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        susp_name = root.attrib.get("reference", "")
        for feat in root.findall("feature"):
            if feat.attrib.get("name") != "plagiarism":
                continue
            src_name = feat.attrib.get("source_reference", "")
            susp_off = int(feat.attrib.get("this_offset", 0))
            susp_len = int(feat.attrib.get("this_length", 0))
            src_off = int(feat.attrib.get("source_offset", 0))
            src_len = int(feat.attrib.get("source_length", 0))
            detections[susp_name].append((
                src_name,
                susp_off, susp_off + susp_len,
                src_off, src_off + src_len,
                feat.attrib.get("obfuscation", "unknown"),
            ))
    return detections


def evaluate_alignment(pred, truth_detections, id2file, file2id):
    """
    Simple alignment evaluation: precision/recall at sentence level.
    pred: list of (susp_start, susp_end, src_doc_id, src_start, src_end, score)
    truth_detections: from build_detections_from_truth
    """
    # This is approximate - real evaluation needs PAN scorer
    total_pred = len(pred)
    total_truth = sum(len(v) for v in truth_detections.values())

    if total_pred == 0 or total_truth == 0:
        return {"prec": 0, "rec": 0, "f1": 0, "pred": total_pred, "truth": total_truth}

    # Count overlapping detections (simplified: Jaccard > 0.3 on character spans)
    hits = 0
    for susp_name, truth_list in truth_detections.items():
        susp_id_opaque = file2id.get(susp_name)
        if not susp_id_opaque:
            continue
        for t_src, t_ss, t_se, t_sos, t_soe, _ in truth_list:
            # Find matching prediction
            for p_ss, p_se, p_src_id, p_sos, p_soe, _ in pred:
                p_susp_name = id2file.get(p_src_id, "")  # this is the source, not susp
                # Check overlap in suspicious document
                overlap_start = max(t_ss, p_ss)
                overlap_end = min(t_se, p_se)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    union = (t_se - t_ss) + (p_se - p_ss) - overlap
                    if union > 0 and overlap / union > 0.3:
                        hits += 1
                        break

    prec = hits / max(total_pred, 1)
    rec = hits / max(total_truth, 1)
    f1 = 2 * prec * rec / max(prec + rec, 0.001)
    return {"prec": prec, "rec": rec, "f1": f1, "pred": total_pred, "truth": total_truth}


def main():
    parser = argparse.ArgumentParser(description="Text alignment for PAN26")
    parser.add_argument("--suspicious", type=Path, required=True,
                        help="queries.jsonl file")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="corpus.jsonl file")
    parser.add_argument("--id-mapping", type=Path,
                        help="id_mapping.tsv from PAN25 conversion")
    parser.add_argument("--run", type=Path,
                        help="TREC run file (which source docs to align against)")
    parser.add_argument("--truth", type=Path,
                        help="qrels.txt for evaluation")
    parser.add_argument("--output", type=Path, default=Path("data/alignment.xml"))
    parser.add_argument("--pan25-xml-dir", type=Path,
                        help="PAN25 truth XML directory for ground truth comparison")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min-block-len", type=int, default=50)
    parser.add_argument("--top-k-sources", type=int, default=3,
                        help="Number of top source docs to align against")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max queries to process (0=all)")
    args = parser.parse_args()

    # Load data
    print(f"Loading corpus from {args.corpus}...")
    corpus = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc.get("doc_id") or doc.get("qid")] = \
                doc.get("default_text") or doc.get("query") or ""

    print(f"Loading suspicious docs from {args.suspicious}...")
    suspicious = {}  # qid -> text
    with open(args.suspicious, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            suspicious[qid] = q.get("query") or q.get("default_text") or ""

    # Load id mapping if available
    id2file, file2id = {}, {}
    if args.id_mapping and args.id_mapping.exists():
        id2file, file2id = load_id_mapping(args.id_mapping)
        print(f"Loaded {len(id2file)} ID mappings")

    # Load truth if available
    truth_detections = {}
    if args.pan25_xml_dir and Path(args.pan25_xml_dir).exists():
        truth_detections = build_detections_from_truth(args.pan25_xml_dir)
        print(f"Loaded truth detections for {len(truth_detections)} docs")

    # Get top-K source docs per query from run file
    run_map = defaultdict(list)
    if args.run and args.run.exists():
        with open(args.run, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                qid, _, doc_id, rank, score, _ = parts
                if int(rank) <= args.top_k_sources:
                    run_map[qid].append((doc_id, float(score)))
        for qid in run_map:
            run_map[qid].sort(key=lambda x: x[1], reverse=True)
        print(f"Loaded run with {len(run_map)} queries")

    # Process
    aligner = Aligner(threshold=args.threshold, min_block_len=args.min_block_len)

    queries = sorted(suspicious.keys())
    if args.max_queries > 0:
        queries = queries[:args.max_queries]

    print(f"\nAligning {len(queries)} queries (threshold={args.threshold})...")
    t0 = time.time()
    all_detections = {}  # qid -> [(susp_start, susp_end, src_doc_id, src_start, src_end, score)]

    for i, qid in enumerate(queries):
        susp_text = suspicious[qid]
        source_texts = []
        for doc_id, score in run_map.get(qid, [])[:args.top_k_sources]:
            if doc_id in corpus:
                source_texts.append((doc_id, corpus[doc_id]))

        if not source_texts:
            continue

        dets = aligner.align_pair(susp_text, source_texts)
        all_detections[qid] = dets

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(queries)} ({elapsed:.1f}s, {elapsed/(i+1):.1f}s per query)")

    elapsed = time.time() - t0
    print(f"Done: {len(queries)} queries in {elapsed:.1f}s")
    print(f"Total detections: {sum(len(v) for v in all_detections.values())}")

    # Evaluate against truth if available
    if truth_detections:
        print("\n=== Alignment Evaluation ===")
        metrics = evaluate_alignment(
            [(s, e, d, ss, se, sc) for dets in all_detections.values()
             for s, e, d, ss, se, sc in dets],
            truth_detections, id2file, file2id
        )
        print(f"Predicted: {metrics['pred']}  Truth: {metrics['truth']}")
        print(f"Precision: {metrics['prec']:.4f}  Recall: {metrics['rec']:.4f}  F1: {metrics['f1']:.4f}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
