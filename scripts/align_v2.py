"""
Text alignment module v2: E5 semantic + Jaccard 3-gram dual filtering + yukino merging.

Pipeline:
  1. Sentence split with char offsets
  2. E5 encode suspicious + source sentences (with query/passage prefixes)
  3. Cosine similarity matrix + Jaccard 3-gram bonus
  4. Threshold detection with dual filters
  5. yukino merging: position proximity + semantic coherence + min length

Usage:
  python scripts/align_v2.py \
    --suspicious data/pan25_retrieval/spot_check/queries.jsonl \
    --corpus data/pan25_retrieval/spot_check/corpus.jsonl \
    --id-mapping data/pan25_retrieval/spot_check/id_mapping.tsv \
    --run data/run_spot_check.txt \
    --pan25-xml-dir pan25/00_spot_check/00_spot_check_truth \
    --output /tmp/align_v2_spotcheck.txt
"""

import argparse
import json
import math
import re
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer


def sent_tokenize(text):
    """Split into sentences with (start, end, text) tracking."""
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|\n\n+|\n(?=[A-Z][a-z])'
    parts = re.split(pattern, text)
    sentences = []
    pos = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        start = text.find(part, pos)
        if start == -1:
            start = pos
        end = start + len(part)
        sentences.append((start, end, part))
        pos = end
    return sentences


def get_3grams(text):
    """Extract character 3-grams for Jaccard."""
    text = text.lower()
    return {text[i:i+3] for i in range(len(text) - 2)}


class AlignerV2:
    def __init__(self, model_name="intfloat/e5-base-v2", threshold=0.70,
                 min_block_len=50, merge_gap=2, jaccard_weight=0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.min_block_len = min_block_len
        self.merge_gap = merge_gap
        self.jaccard_weight = jaccard_weight
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = 512
        return self._model

    def encode(self, texts, prefix="passage: "):
        if not texts:
            return np.array([])
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )

    def align_pair(self, susp_text, source_texts):
        """
        Align suspicious doc against top-K source docs.
        Returns detections: [(susp_start, susp_end, src_doc_id, src_start, src_end, score), ...]
        """
        susp_sents = sent_tokenize(susp_text)
        if len(susp_sents) < 2:
            return []

        susp_emb = self.encode([s[2] for s in susp_sents], prefix="query: ")
        susp_3grams = [get_3grams(s[2]) for s in susp_sents]

        all_detections = []

        for src_doc_id, src_text in source_texts[:5]:
            src_sents = sent_tokenize(src_text)
            if len(src_sents) < 2:
                continue

            src_emb = self.encode([s[2] for s in src_sents], prefix="passage: ")
            src_3grams = [get_3grams(s[2]) for s in src_sents]

            # ---- Dual scoring: cosine + Jaccard ----
            cos_sim = susp_emb @ src_emb.T  # (n_susp, n_src)

            # Jaccard 3-gram matrix (expensive, compute lazily)
            jaccard_bonus = np.zeros_like(cos_sim)
            # Only compute top candidates per suspicious sentence for speed
            top_k_per_susp = min(20, len(src_sents))
            for i in range(len(susp_sents)):
                top_src = np.argpartition(cos_sim[i], -top_k_per_susp)[-top_k_per_susp:]
                for j in top_src:
                    sg = susp_3grams[i]
                    dg = src_3grams[j]
                    if sg and dg:
                        jaccard_bonus[i, j] = len(sg & dg) / max(len(sg | dg), 1)

            # Final score: cos * (1 + w * jaccard)
            final_score = cos_sim * (1.0 + self.jaccard_weight * jaccard_bonus)

            # Best source sentence per suspicious sentence
            best_src_idx = np.argmax(final_score, axis=1)
            best_scores = np.max(final_score, axis=1)

            # ---- Bidirectional matching filter (relaxed: ±5 window) ----
            best_susp_idx = np.argmax(final_score, axis=0)  # best susp per src
            bidir_ok = np.zeros(len(susp_sents), dtype=bool)
            for i in range(len(susp_sents)):
                j = best_src_idx[i]
                if best_scores[i] >= self.threshold:
                    if abs(best_susp_idx[j] - i) <= 5:
                        bidir_ok[i] = True

            # Mark sentences: threshold only (no length ratio)
            marked = bidir_ok
            if not marked.any():
                continue

            # ---- yukino merging ----
            blocks = self._yukino_merge(
                susp_sents, src_sents, marked, best_src_idx,
                best_scores, src_doc_id, final_score
            )
            all_detections.extend(blocks)

        all_detections.sort(key=lambda d: d[0])
        return all_detections

    def _yukino_merge(self, susp_sents, src_sents, marked, best_src_idx, best_scores, src_doc_id, sim_matrix):
        """
        Three yukino conditions for merging adjacent sentences:
          1. Position proximity: gap <= merge_gap sentences
          2. Semantic coherence: std of block cosine scores < 0.15
          3. Minimum length: block >= min_block_len characters
        """
        blocks = []
        i = 0
        while i < len(marked):
            if not marked[i]:
                i += 1
                continue

            block_start = i
            block_scores = [best_scores[i]]
            block_src_indices = [best_src_idx[i]]

            j = i + 1
            while j < len(marked):
                if not marked[j]:
                    j += 1
                    continue

                # Condition 1: position proximity
                gap = j - (block_start + len(block_src_indices) - 1) - 1
                if gap > self.merge_gap:
                    break

                # Check same source (nearby chunks = same document region)
                last_src = block_src_indices[-1]
                if abs(best_src_idx[j] - last_src) > 5:
                    j += 1
                    continue

                # Condition 2: semantic coherence check
                # Compute mean cosine between last 3 and new sentence
                test_scores = block_scores[-3:] + [best_scores[j]]
                if np.std(test_scores) > 0.15:
                    j += 1
                    continue

                block_scores.append(best_scores[j])
                block_src_indices.append(best_src_idx[j])
                j += 1

            block_end = j

            # Condition 3: minimum length
            char_len = sum(
                susp_sents[k][1] - susp_sents[k][0]
                for k in range(block_start, min(block_end, len(susp_sents)))
            )
            if char_len < self.min_block_len:
                i = block_end
                continue

            # Source offsets from best-matching source sentences
            src_start = min(
                src_sents[idx][0] for idx in block_src_indices if idx < len(src_sents)
            )
            src_end = max(
                src_sents[idx][1] for idx in block_src_indices if idx < len(src_sents)
            )

            susp_start = susp_sents[block_start][0]
            susp_end = susp_sents[min(block_end - 1, len(susp_sents) - 1)][1]
            avg_score = float(np.mean(block_scores))

            blocks.append((susp_start, susp_end, src_doc_id, src_start, src_end, avg_score))
            i = block_end

        return blocks


def build_truth_from_xml(truth_dir):
    """Parse PAN25 truth XML → {susp_filename: [(src_filename, s_start, s_end, src_start, src_end, obfuscation)]}"""
    import xml.etree.ElementTree as ET
    detections = defaultdict(list)
    for xml_path in sorted(Path(truth_dir).glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        susp_name = root.attrib.get("reference", "")
        for feat in root.findall("feature"):
            if feat.attrib.get("name") != "plagiarism":
                continue
            detections[susp_name].append((
                feat.attrib.get("source_reference", ""),
                int(feat.attrib.get("this_offset", 0)),
                int(feat.attrib.get("this_offset", 0)) + int(feat.attrib.get("this_length", 0)),
                int(feat.attrib.get("source_offset", 0)),
                int(feat.attrib.get("source_offset", 0)) + int(feat.attrib.get("source_length", 0)),
                feat.attrib.get("obfuscation", ""),
            ))
    return detections


def evaluate_per_query(all_preds, truth, id2file, file2id):
    """Per-query character-level overlap evaluation."""
    if not all_preds or not truth:
        total_truth = sum(len(v) for v in truth.values())
        return {"prec": 0, "rec": 0, "f1": 0, "pred": sum(len(v) for v in all_preds.values()), "truth": total_truth}

    total_pred = 0
    total_truth = 0
    total_hits = 0

    for qid, preds in all_preds.items():
        susp_file = id2file.get(qid, "")
        if susp_file not in truth:
            continue
        truth_list = truth[susp_file]
        total_pred += len(preds)
        total_truth += len(truth_list)

        for t_src, t_ss, t_se, t_sos, t_soe, _ in truth_list:
            for p_ss, p_se, p_src_id, p_sos, p_soe, _ in preds:
                overlap_s = max(t_ss, p_ss)
                overlap_e = min(t_se, p_se)
                if overlap_e > overlap_s:
                    overlap = overlap_e - overlap_s
                    union = (t_se - t_ss) + (p_se - p_ss) - overlap
                    if union > 0 and overlap / union > 0.3:
                        total_hits += 1
                        break

    prec = total_hits / max(total_pred, 1)
    rec = total_hits / max(total_truth, 1)
    f1 = 2 * prec * rec / max(prec + rec, 0.001)
    return {"prec": prec, "rec": rec, "f1": f1, "pred": total_pred, "truth": total_truth, "hits": total_hits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suspicious", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--id-mapping", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--pan25-xml-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/align_v2.txt"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--jaccard-weight", type=float, default=0.5)
    parser.add_argument("--min-block-len", type=int, default=50)
    parser.add_argument("--merge-gap", type=int, default=2)
    parser.add_argument("--top-k-sources", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    # Load data
    print(f"Loading corpus...")
    corpus = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            corpus[d.get("doc_id") or d.get("qid")] = d.get("default_text") or ""

    print(f"Loading suspicious...")
    suspicious = {}
    with open(args.suspicious, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            suspicious[q.get("qid") or q.get("query_id")] = q.get("query") or q.get("default_text") or ""

    # ID mapping
    id2file, file2id = {}, {}
    if args.id_mapping and args.id_mapping.exists():
        with open(args.id_mapping) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    id2file[parts[2]] = parts[3]
                    file2id[parts[3]] = parts[2]

    # Truth (lazy load)
    truth = {}
    truth_dir = args.pan25_xml_dir

    # Run file
    run = defaultdict(list)
    if args.run and args.run.exists():
        with open(args.run) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    run[parts[0]].append((parts[2], float(parts[4])))
        for qid in run:
            run[qid].sort(key=lambda x: x[1], reverse=True)

    # Align
    aligner = AlignerV2(
        threshold=args.threshold,
        jaccard_weight=args.jaccard_weight,
        min_block_len=args.min_block_len,
        merge_gap=args.merge_gap,
    )

    queries = sorted(suspicious.keys())
    if args.max_queries > 0:
        queries = queries[:args.max_queries]

    print(f"\nAligning {len(queries)} queries (th={args.threshold}, jw={args.jaccard_weight})...")
    t0 = time.time()
    all_preds = {}  # qid -> [(s,e,src_id,src_s,src_e,score), ...]
    for i, qid in enumerate(queries):
        susp_text = suspicious[qid]
        srcs = [(d, corpus.get(d, "")) for d, _ in run.get(qid, [])[:args.top_k_sources] if d in corpus]
        if not srcs:
            continue
        dets = aligner.align_pair(susp_text, srcs)
        all_preds[qid] = dets
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(queries)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    total_dets = sum(len(v) for v in all_preds.values())
    print(f"Done: {len(queries)} queries in {elapsed:.1f}s, {total_dets} detections")

    if truth_dir and Path(truth_dir).exists():
        truth = build_truth_from_xml(truth_dir)
        print(f"Truth: {len(truth)} docs, {sum(len(v) for v in truth.values())} segments")
        m = evaluate_per_query(all_preds, truth, id2file, file2id)
        print(f"\nEval: Prec={m['prec']:.4f} Rec={m['rec']:.4f} F1={m['f1']:.4f}")
        print(f"  Pred={m['pred']} Truth={m['truth']} Hits={m.get('hits', '?')}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
