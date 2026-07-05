"""
Convert PAN25 Generated Plagiarism Detection data to PAN26-style retrieval data.

Outputs one directory per split under data/pan25_retrieval/:
  corpus.jsonl      source documents, with opaque doc_id values
  corpus.jsonl.gz   gzipped copy of corpus.jsonl
  queries.jsonl     suspicious documents that have at least one positive source
  qrels.txt         TREC qrels: qid 0 doc_id 1
  id_mapping.tsv    opaque IDs back to original filenames for debugging only

Only XML features with name="plagiarism" are treated as relevance judgments.
Files with only name="altered" are skipped because they do not identify a
source document for retrieval.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAN25_DIR = PROJECT_ROOT / "pan25"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "pan25_retrieval"
LOCK_PATH = PROJECT_ROOT / "data" / ".convert_pan25.lock"

SPLITS = {
    "spot_check": {
        "data": PAN25_DIR / "00_spot_check" / "00_spot_check",
        "truth": PAN25_DIR / "00_spot_check" / "00_spot_check_truth",
    },
    "train": {
        "data": PAN25_DIR / "01_train" / "01_train",
        "truth": PAN25_DIR / "01_train" / "01_train_truth",
    },
    "holdout": {
        "data": PAN25_DIR / "02_validation" / "02_validation",
        "truth": PAN25_DIR / "02_validation" / "02_validation_truth",
    },
}


def stable_id(prefix: str, split: str, filename: str) -> str:
    digest = hashlib.sha1(f"pan25:{split}:{filename}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_truth(truth_dir: Path) -> dict[str, set[str]]:
    qrels_by_susp: dict[str, set[str]] = defaultdict(set)

    for xml_path in sorted(truth_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        susp_name = root.attrib.get("reference")
        if not susp_name:
            continue

        for feature in root.findall("feature"):
            if feature.attrib.get("name") != "plagiarism":
                continue

            source_name = feature.attrib.get("source_reference")
            if source_name:
                qrels_by_susp[susp_name].add(source_name)

    return qrels_by_susp


def write_jsonl_row(out, row: dict) -> None:
    out.write(json.dumps(row, ensure_ascii=False) + "\n")


def gzip_copy(src: Path, dst: Path) -> None:
    with src.open("rb") as source, gzip.open(dst, "wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)


def convert_split(split: str, data_dir: Path, truth_dir: Path, output_dir: Path) -> dict[str, int]:
    susp_dir = data_dir / "susp"
    src_dir = data_dir / "src"
    qrels_by_susp = parse_truth(truth_dir)

    source_files = sorted(src_dir.glob("*.txt"))
    source_ids = {path.name: stable_id("d", split, path.name) for path in source_files}

    missing_susp = 0
    missing_source = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as out:
        for path in source_files:
            write_jsonl_row(out, {"doc_id": source_ids[path.name], "default_text": read_text(path)})

    query_count = 0
    qrel_count = 0
    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    with queries_path.open("w", encoding="utf-8") as queries_out, qrels_path.open(
        "w", encoding="utf-8"
    ) as qrels_out:
        for susp_name in sorted(qrels_by_susp):
            susp_path = susp_dir / susp_name
            if not susp_path.exists():
                missing_susp += 1
                continue

            source_names = sorted(qrels_by_susp[susp_name])
            existing_sources = [name for name in source_names if name in source_ids]
            missing_source += len(source_names) - len(existing_sources)
            if not existing_sources:
                continue

            qid = stable_id("q", split, susp_name)
            write_jsonl_row(queries_out, {"qid": qid, "query": read_text(susp_path)})
            query_count += 1
            for source_name in existing_sources:
                qrels_out.write(f"{qid} 0 {source_ids[source_name]} 1\n")
                qrel_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    gzip_copy(corpus_path, output_dir / "corpus.jsonl.gz")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "id_mapping.tsv").open("w", encoding="utf-8") as out:
        out.write("kind\tsplit\topaque_id\tfilename\n")
        for susp_name in sorted(qrels_by_susp):
            out.write(f"query\t{split}\t{stable_id('q', split, susp_name)}\t{susp_name}\n")
        for source_name, doc_id in sorted(source_ids.items()):
            out.write(f"doc\t{split}\t{doc_id}\t{source_name}\n")

    return {
        "source_docs": len(source_files),
        "queries": query_count,
        "qrels": qrel_count,
        "missing_susp": missing_susp,
        "missing_source": missing_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Another conversion appears to be running: {LOCK_PATH}")

    os.write(lock_fd, str(os.getpid()).encode("utf-8"))
    os.close(lock_fd)
    atexit.register(lambda: LOCK_PATH.exists() and LOCK_PATH.unlink())

    for split, paths in SPLITS.items():
        stats = convert_split(split, paths["data"], paths["truth"], args.output_root / split)
        print(
            f"{split}: "
            f"{stats['source_docs']} source docs, "
            f"{stats['queries']} queries, "
            f"{stats['qrels']} qrels"
        , flush=True)
        if stats["missing_susp"] or stats["missing_source"]:
            print(
                f"  skipped missing files: "
                f"{stats['missing_susp']} suspicious, {stats['missing_source']} sources"
            )


if __name__ == "__main__":
    main()
