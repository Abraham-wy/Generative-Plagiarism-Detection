"""
Convert PAN14 source-retrieval corpus to PAN26 format (queries.jsonl, qrels.txt, missing_sources.tsv).

PAN14 corpus:
  suspicious-documentNNN-batch1.{txt,html,json}
  - .txt  -> plain text of suspicious document
  - .json -> metadata with plagiarism source URLs (ChatNoir/ClueWeb09)
  - .html -> color-coded HTML version

Source URLs point to an old ChatNoir instance. We extract:
  - ClueWeb09 doc IDs (from ?id=XXX)
  - Source page URLs (from ?href=XXX)
These need to be fetched via the new ChatNoir API.

Output:
  queries.jsonl   - suspicious documents as queries
  qrels.txt       - query_id -> source_doc_id relevance mapping
  missing_sources.tsv - source URLs/IDs that need fetching
"""

import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs
from collections import defaultdict

PROJECT_ROOT = Path("/Users/wy/Library/Mobile Documents/com~apple~CloudDocs/2026/科研/gpdetection-test")
PAN14_DIR = PROJECT_ROOT / "pan14-source-retrieval-training-corpus-2014-12-01"
DATA_DIR = PROJECT_ROOT / "data" / "pan14"


def read_text_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_source_url(raw_url):
    """
    Parse a ChatNoir source URL and extract the relevant identifier.

    Two formats:
    1. ?id=CLUEWEB_DOC_ID  -> ("clueweb_id", "00968616391")
    2. ?href=ENCODED_URL   -> ("source_url", "http://example.com/page")
    """
    if "?id=" in raw_url:
        parsed = parse_qs(urlparse(raw_url).query)
        doc_id = parsed.get("id", [None])[0]
        return ("clueweb_id", doc_id)
    elif "?href=" in raw_url:
        parsed = parse_qs(urlparse(raw_url).query)
        href = parsed.get("href", [None])[0]
        if href:
            href = unquote(href)
        return ("source_url", href)
    else:
        return ("unknown", raw_url)


def extract_doc_number(filename):
    """Extract document number from filename like 'suspicious-document005-batch1.txt'"""
    m = re.search(r'suspicious-document(\d+)-batch', filename)
    if m:
        return int(m.group(1))
    return None


def process_pan14():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(
        [f for f in os.listdir(PAN14_DIR) if f.endswith('.json')],
        key=lambda x: extract_doc_number(x) or 0
    )

    print(f"Found {len(json_files)} JSON metadata files")

    queries = []
    qrels = []
    missing_sources = {}  # source_id -> {"type": ..., "value": ..., "query_ids": [...]}
    source_id_counter = 0
    total_plagiarism_segments = 0
    skipped_json = 0

    for i, json_file in enumerate(json_files):
        json_path = PAN14_DIR / json_file

        # Parse JSON with error handling
        with open(json_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        try:
            metadata = json.loads(content)
        except json.JSONDecodeError:
            # Fix trailing commas (common in PAN14 JSON files)
            import re as _re
            fixed = _re.sub(r',\s*}', '}', content)
            fixed = _re.sub(r',\s*]', ']', fixed)
            try:
                metadata = json.loads(fixed)
            except json.JSONDecodeError as e:
                print(f"  WARN: JSON parse error in {json_file}: {e}")
                skipped_json += 1
                continue

        susp_html_name = metadata.get("suspicious-document", "")
        # Derive basename: suspicious-document005-batch1
        basename = susp_html_name.replace(".html", "")
        txt_file = basename + ".txt"
        txt_path = PAN14_DIR / txt_file

        if not txt_path.exists():
            # Try without -batch1 suffix
            print(f"  WARN: missing txt for {basename}")
            continue

        susp_text = read_text_file(txt_path)
        query_id = basename

        # Add query
        queries.append({
            "query_id": query_id,
            "default_text": susp_text,
            "metadata": {
                "source": "pan14",
                "language": metadata.get("language", ""),
            }
        })

        # Process plagiarism sources
        plagiarism_segments = metadata.get("plagiarism", [])
        seen_sources = set()  # deduplicate sources for this query

        for seg in plagiarism_segments:
            total_plagiarism_segments += 1
            source_url = seg.get("source-url", "")
            if not source_url:
                continue

            src_type, src_value = parse_source_url(source_url)

            if src_type == "unknown":
                print(f"  WARN: unknown source URL format: {source_url}")
                continue

            # Create a unique source identifier
            source_key = f"{src_type}:{src_value}"

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                source_id_counter += 1

                source_doc_id = f"pan14-src-{source_id_counter:06d}"

                if source_key not in missing_sources:
                    missing_sources[source_key] = {
                        "type": src_type,
                        "value": src_value,
                        "source_doc_id": source_doc_id,
                    }

                # qrels: query_id 0 doc_id relevance
                qrels.append(f"{query_id} 0 {source_doc_id} 1")

        if (i + 1) % 20 == 0:
            print(f"  ... processed {i + 1}/{len(json_files)}")

    print(f"\nProcessed {len(json_files) - skipped_json} documents ({skipped_json} skipped)")
    print(f"Total plagiarism segments: {total_plagiarism_segments}")
    print(f"Unique source URLs to fetch: {len(missing_sources)}")

    # Count by type
    id_count = sum(1 for v in missing_sources.values() if v["type"] == "clueweb_id")
    url_count = sum(1 for v in missing_sources.values() if v["type"] == "source_url")
    print(f"  ClueWeb09 IDs: {id_count}")
    print(f"  Source URLs: {url_count}")

    # Write queries.jsonl
    queries_path = DATA_DIR / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(queries)} queries to {queries_path}")

    # Write qrels.txt
    qrels_path = DATA_DIR / "qrels.txt"
    with open(qrels_path, "w", encoding="utf-8") as f:
        for line in qrels:
            f.write(line + "\n")
    print(f"Wrote {len(qrels)} qrels to {qrels_path}")

    # Write missing_sources.tsv
    missing_path = DATA_DIR / "missing_sources.tsv"
    with open(missing_path, "w", encoding="utf-8") as f:
        f.write("source_key\ttype\tvalue\tsource_doc_id\n")
        for key, info in sorted(missing_sources.items()):
            f.write(f"{key}\t{info['type']}\t{info['value']}\t{info['source_doc_id']}\n")
    print(f"Wrote {len(missing_sources)} missing sources to {missing_path}")

    return queries, qrels, missing_sources


if __name__ == "__main__":
    process_pan14()
