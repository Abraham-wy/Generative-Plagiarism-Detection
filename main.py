#!/usr/bin/env python3
"""
PAN text-alignment baseline for generated plagiarism detection.

Input layout:
    /input/
      pairs
      susp/<suspicious-document>.txt
      src/<source-document>.txt

Execution:
    python main.py /input /output

For each pair, the system writes one PAN-style XML file:
    <suspicious-stem>-<source-stem>.xml

The implementation is intentionally self-contained at runtime: no PyTerrier,
Java, external search service, or API call is used. It performs indexless
pairwise semantic matching with sentence-transformer embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np


LOG = logging.getLogger("pan-baseline")


@dataclass(frozen=True)
class TextSpan:
    text: str
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class Detection:
    this_offset: int
    this_length: int
    source_reference: str
    source_offset: int
    source_length: int
    score: float

    @property
    def this_end(self) -> int:
        return self.this_offset + self.this_length

    @property
    def source_end(self) -> int:
        return self.source_offset + self.source_length


def read_text(path: Path) -> str:
    """Read PAN text files while tolerating common corpus encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def parse_pairs(pairs_path: Path) -> list[tuple[str, str]]:
    """Parse a PAN pairs file with whitespace, comma, semicolon, or tab separators."""
    pairs: list[tuple[str, str]] = []
    for line_no, raw_line in enumerate(pairs_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in re.split(r"[\s,;]+", line) if part]
        if len(parts) < 2:
            raise ValueError(f"{pairs_path}:{line_no}: expected two filenames, got {raw_line!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def find_document(dataset_dir: Path, subdir: str, filename: str) -> Path:
    """Find a document in the expected subdir, with a recursive fallback."""
    direct = dataset_dir / subdir / filename
    if direct.exists():
        return direct

    matches = list((dataset_dir / subdir).rglob(filename)) if (dataset_dir / subdir).exists() else []
    if not matches:
        matches = list(dataset_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {dataset_dir}/{subdir}")
    return matches[0]


def sentence_spans(text: str) -> list[TextSpan]:
    """
    Split text into sentence-ish spans while preserving character offsets.

    This avoids NLTK/spaCy downloads and works offline. For scientific text, it
    is deliberately conservative: punctuation followed by whitespace and a
    capital/number/section marker starts a new span.
    """
    spans: list[TextSpan] = []
    pattern = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9#*\-\"'(\[])")
    start = 0
    for match in pattern.finditer(text):
        end = match.start()
        _append_nonempty_span(spans, text, start, end)
        start = match.end()
    _append_nonempty_span(spans, text, start, len(text))

    if spans:
        return spans

    stripped_start = len(text) - len(text.lstrip())
    stripped = text.strip()
    return [TextSpan(stripped, stripped_start, len(stripped))] if stripped else []


def _append_nonempty_span(spans: list[TextSpan], text: str, start: int, end: int) -> None:
    raw = text[start:end]
    left_trimmed = len(raw) - len(raw.lstrip())
    clean = raw.strip()
    if clean:
        offset = start + left_trimmed
        spans.append(TextSpan(clean, offset, len(clean)))


def make_windows(
    text: str,
    *,
    window_chars: int,
    step_sentences: int,
    min_chars: int,
) -> list[TextSpan]:
    """Build overlapping sentence windows with char offsets."""
    sentences = sentence_spans(text)
    if not sentences:
        return []

    windows: list[TextSpan] = []
    idx = 0
    while idx < len(sentences):
        start = sentences[idx].offset
        end = sentences[idx].end
        j = idx + 1
        while j < len(sentences) and (sentences[j].end - start) <= window_chars:
            end = sentences[j].end
            j += 1

        if end - start >= min_chars or idx == 0:
            chunk = text[start:end].strip()
            if chunk:
                left_trimmed = len(text[start:end]) - len(text[start:end].lstrip())
                offset = start + left_trimmed
                windows.append(TextSpan(chunk, offset, len(chunk)))

        idx += max(1, step_sentences)

    # Very short documents still need at least one comparable unit.
    if not windows:
        clean = text.strip()
        if clean:
            offset = len(text) - len(text.lstrip())
            windows.append(TextSpan(clean, offset, len(clean)))

    return windows


def embed_spans(model: Any, spans: list[TextSpan], batch_size: int) -> np.ndarray:
    if not spans:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        [span.text for span in spans],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def detect_pair(
    model: Any,
    susp_text: str,
    src_text: str,
    source_reference: str,
    *,
    threshold: float,
    window_chars: int,
    step_sentences: int,
    min_chars: int,
    batch_size: int,
    top_k_per_susp_window: int,
    merge_gap: int,
    max_detections: int,
) -> list[Detection]:
    susp_windows = make_windows(
        susp_text,
        window_chars=window_chars,
        step_sentences=step_sentences,
        min_chars=min_chars,
    )
    src_windows = make_windows(
        src_text,
        window_chars=window_chars,
        step_sentences=step_sentences,
        min_chars=min_chars,
    )
    if not susp_windows or not src_windows:
        return []

    susp_emb = embed_spans(model, susp_windows, batch_size)
    src_emb = embed_spans(model, src_windows, batch_size)
    similarities = susp_emb @ src_emb.T

    candidates: list[Detection] = []
    for susp_idx, row in enumerate(similarities):
        if top_k_per_susp_window >= len(row):
            best_src_indices = np.argsort(row)[::-1]
        else:
            best_src_indices = np.argpartition(row, -top_k_per_susp_window)[-top_k_per_susp_window:]
            best_src_indices = best_src_indices[np.argsort(row[best_src_indices])[::-1]]

        for src_idx in best_src_indices:
            score = float(row[src_idx])
            if score < threshold:
                continue
            susp_span = susp_windows[susp_idx]
            src_span = src_windows[int(src_idx)]
            candidates.append(
                Detection(
                    this_offset=susp_span.offset,
                    this_length=susp_span.length,
                    source_reference=source_reference,
                    source_offset=src_span.offset,
                    source_length=src_span.length,
                    score=score,
                )
            )

    return merge_detections(candidates, merge_gap=merge_gap, max_detections=max_detections)


def merge_detections(
    detections: Iterable[Detection],
    *,
    merge_gap: int,
    max_detections: int,
) -> list[Detection]:
    """Merge nearby same-source hits into larger PAN-style passages."""
    ordered = sorted(
        detections,
        key=lambda det: (det.this_offset, det.source_offset, -det.score),
    )
    merged: list[Detection] = []

    for det in ordered:
        if not merged:
            merged.append(det)
            continue

        prev = merged[-1]
        same_source = det.source_reference == prev.source_reference
        suspicious_close = det.this_offset <= prev.this_end + merge_gap
        source_close = det.source_offset <= prev.source_end + merge_gap

        if same_source and suspicious_close and source_close:
            prev_this_end = max(prev.this_end, det.this_end)
            prev_source_end = max(prev.source_end, det.source_end)
            prev.this_length = prev_this_end - prev.this_offset
            prev.source_length = prev_source_end - prev.source_offset
            prev.score = max(prev.score, det.score)
        elif not _is_covered(det, merged):
            merged.append(det)

    merged = sorted(merged, key=lambda det: (-det.score, det.this_offset, det.source_offset))
    return merged[:max_detections]


def _is_covered(det: Detection, existing: list[Detection]) -> bool:
    for prev in existing:
        if prev.source_reference != det.source_reference:
            continue
        suspicious_covered = det.this_offset >= prev.this_offset and det.this_end <= prev.this_end
        source_covered = det.source_offset >= prev.source_offset and det.source_end <= prev.source_end
        if suspicious_covered and source_covered:
            return True
    return False


def write_xml(output_path: Path, suspicious_reference: str, detections: list[Detection]) -> None:
    root = Element("document", {"reference": suspicious_reference})
    for det in sorted(detections, key=lambda item: (item.this_offset, item.source_offset)):
        SubElement(
            root,
            "feature",
            {
                "name": "detected-plagiarism",
                "this_offset": str(int(det.this_offset)),
                "this_length": str(int(det.this_length)),
                "source_reference": det.source_reference,
                "source_offset": str(int(det.source_offset)),
                "source_length": str(int(det.source_length)),
            },
        )
    tree = ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def write_jsonl(output_path: Path, suspicious_reference: str, detections: list[Detection]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for det in sorted(detections, key=lambda item: (item.this_offset, item.source_offset)):
            f.write(
                json.dumps(
                    {
                        "document": suspicious_reference,
                        "name": "detected-plagiarism",
                        "this_offset": det.this_offset,
                        "this_length": det.this_length,
                        "source_reference": det.source_reference,
                        "source_offset": det.source_offset,
                        "source_length": det.source_length,
                        "score": round(det.score, 6),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def pair_output_stem(susp_name: str, src_name: str) -> str:
    return f"{Path(susp_name).stem}-{Path(src_name).stem}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline PAN plagiarism text-alignment baseline.")
    parser.add_argument("input_dir", type=Path, help="Dataset directory containing pairs, susp/, and src/.")
    parser.add_argument("output_dir", type=Path, help="Directory where XML/JSONL outputs will be written.")
    parser.add_argument("--model", default=os.environ.get("PAN_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("PAN_THRESHOLD", "0.80")))
    parser.add_argument("--window-chars", type=int, default=int(os.environ.get("PAN_WINDOW_CHARS", "550")))
    parser.add_argument("--step-sentences", type=int, default=int(os.environ.get("PAN_STEP_SENTENCES", "2")))
    parser.add_argument("--min-chars", type=int, default=int(os.environ.get("PAN_MIN_CHARS", "80")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PAN_BATCH_SIZE", "32")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("PAN_TOP_K", "2")))
    parser.add_argument("--merge-gap", type=int, default=int(os.environ.get("PAN_MERGE_GAP", "120")))
    parser.add_argument("--max-detections", type=int, default=int(os.environ.get("PAN_MAX_DETECTIONS", "200")))
    parser.add_argument(
        "--format",
        choices=("xml", "jsonl"),
        default=os.environ.get("PAN_OUTPUT_FORMAT", "xml"),
        help="PAN XML is the default; JSONL is provided for easier debugging.",
    )
    parser.add_argument("--log-level", default=os.environ.get("PAN_LOG_LEVEL", "INFO"))
    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    pairs_path = args.input_dir / "pairs"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs file: {pairs_path}")

    pairs = parse_pairs(pairs_path)
    LOG.info("Loaded %d document pairs from %s", len(pairs), pairs_path)
    LOG.info("Loading sentence-transformer model: %s", args.model)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.model))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (susp_name, src_name) in enumerate(pairs, start=1):
        susp_path = find_document(args.input_dir, "susp", susp_name)
        src_path = find_document(args.input_dir, "src", src_name)
        LOG.info("[%d/%d] %s vs %s", idx, len(pairs), susp_name, src_name)

        detections = detect_pair(
            model,
            read_text(susp_path),
            read_text(src_path),
            src_name,
            threshold=args.threshold,
            window_chars=args.window_chars,
            step_sentences=args.step_sentences,
            min_chars=args.min_chars,
            batch_size=args.batch_size,
            top_k_per_susp_window=args.top_k,
            merge_gap=args.merge_gap,
            max_detections=args.max_detections,
        )

        stem = pair_output_stem(susp_name, src_name)
        if args.format == "xml":
            out_path = args.output_dir / f"{stem}.xml"
            write_xml(out_path, susp_name, detections)
        else:
            out_path = args.output_dir / f"{stem}.jsonl"
            write_jsonl(out_path, susp_name, detections)
        LOG.info("Wrote %s (%d detections)", out_path, len(detections))

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        LOG.exception("Baseline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
