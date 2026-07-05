"""
Fetch source document texts for PAN14 missing sources.

Two source types:
  1. source_url → try direct HTTP, then Wayback Machine
  2. clueweb_id → ChatNoir API (requires CHATNOIR_API_KEY env var)

Outputs:
  data/pan14/corpus.jsonl  - fetched source documents
  data/pan14/fetch_errors.tsv - failed fetches

Usage:
  # Without ChatNoir key (only source_url types will be fetched):
  python scripts/fetch_sources.py

  # With ChatNoir key:
  CHATNOIR_API_KEY=your-key python scripts/fetch_sources.py
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/Users/wy/Library/Mobile Documents/com~apple~CloudDocs/2026/科研/gpdetection-test"))
DATA_DIR = PROJECT_ROOT / "data" / "pan14"
MISSING_SOURCES = DATA_DIR / "missing_sources.tsv"
CORPUS_OUT = DATA_DIR / "corpus.jsonl"
ERRORS_OUT = DATA_DIR / "fetch_errors.tsv"

CHATNOIR_API_KEY = os.environ.get("CHATNOIR_API_KEY", "")

TIMEOUT = 30
MAX_RETRIES = 2
REQUEST_DELAY = 0.3


def clean_html(text):
    """Strip HTML to plain text."""
    if not text:
        return text
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_url(url):
    """Fetch a URL, returning (text, source_description)."""
    ua = "Mozilla/5.0 (compatible; PAN14-research-bot/1.0)"
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                return text, "direct"
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY)
    return None, "direct_failed"


def fetch_wayback(url):
    """Try Wayback Machine snapshot from around 2014."""
    wayback_url = f"https://web.archive.org/web/20140101000000id_/{url}"
    return fetch_url(wayback_url)


def fetch_chatnoir(doc_id):
    """Fetch via ChatNoir API by ClueWeb09 document ID."""
    import urllib.request as req

    url = f"https://www.chatnoir.eu/api/v1/_search"
    query = {
        "query": {"bool": {"filter": [{"term": {"warc_header_document_id": doc_id}}]}},
        "size": 1,
        "_source": ["body", "plain", "title"],
    }
    data = json.dumps(query).encode("utf-8")

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = req.Request(url, data=data, headers={
                "Content-Type": "application/json",
                "X-API-Key": CHATNOIR_API_KEY,
            })
            with req.urlopen(r, timeout=TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                hits = result.get("hits", {}).get("hits", [])
                if hits:
                    src = hits[0].get("_source", {})
                    body = src.get("body") or src.get("plain") or json.dumps(src)
                    return body, "chatnoir"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY)
    return None, "chatnoir_failed"


def fetch_one(entry):
    """Fetch one source. Returns (source_doc_id, text, source_type, fetch_method, error)."""
    src_type = entry["type"]
    src_value = entry["value"]
    src_doc_id = entry["source_doc_id"]

    if src_type == "source_url":
        # Try direct first, then Wayback
        text, method = fetch_url(src_value)
        if not text:
            text, wb_method = fetch_wayback(src_value)
            method = f"wayback({wb_method})" if text else "all_failed"
        return src_doc_id, clean_html(text) if text else "", src_type, method, None if text else "unreachable"

    elif src_type == "clueweb_id":
        if not CHATNOIR_API_KEY:
            return src_doc_id, "", src_type, "skipped_no_key", "CHATNOIR_API_KEY not set"
        text, method = fetch_chatnoir(src_value)
        return src_doc_id, clean_html(text) if text else "", src_type, method, None if text else "chatnoir_no_result"

    return src_doc_id, "", src_type, "unknown", "unknown source type"


def main():
    if not MISSING_SOURCES.exists():
        print(f"ERROR: {MISSING_SOURCES} not found. Run convert_pan14.py first.")
        return

    # Read sources
    sources = []
    with open(MISSING_SOURCES, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                sources.append({
                    "source_key": parts[0],
                    "type": parts[1],
                    "value": parts[2],
                    "source_doc_id": parts[3],
                })

    print(f"Loaded {len(sources)} sources to fetch")
    print(f"  ClueWeb09 IDs: {sum(1 for s in sources if s['type'] == 'clueweb_id')}")
    print(f"  Source URLs: {sum(1 for s in sources if s['type'] == 'source_url')}")
    print(f"  ChatNoir API key: {'configured' if CHATNOIR_API_KEY else 'NOT SET'}")
    print()

    # Fetch
    fetched = []
    errors = []
    total = len(sources)
    workers = min(8, total)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, s): s for s in sources}
        for i, future in enumerate(as_completed(futures), 1):
            doc_id, text, stype, method, error = future.result()
            if text and len(text.strip()) > 50:
                fetched.append((doc_id, text))
            else:
                errors.append({
                    "source_doc_id": doc_id,
                    "type": stype,
                    "value": futures[future]["value"],
                    "method": method,
                    "error": error or "content too short or empty",
                })
            if i % 100 == 0:
                print(f"  {i}/{total} ({len(fetched)} ok, {len(errors)} fail)")

    print(f"\nDone: {len(fetched)} fetched, {len(errors)} failed")

    # Write corpus.jsonl
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        for doc_id, text in fetched:
            f.write(json.dumps({"doc_id": doc_id, "default_text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {CORPUS_OUT} ({len(fetched)} docs)")

    # Write errors
    if errors:
        with open(ERRORS_OUT, "w", encoding="utf-8") as f:
            f.write("source_doc_id\ttype\tvalue\tmethod\terror\n")
            for e in errors:
                f.write(f"{e['source_doc_id']}\t{e['type']}\t{e['value']}\t{e['method']}\t{e['error']}\n")
        print(f"Wrote {ERRORS_OUT} ({len(errors)} errors)")


if __name__ == "__main__":
    main()
