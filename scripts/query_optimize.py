"""
Query representation optimization: replace raw first-2000-char truncation
with smarter extraction strategies.

Strategies:
  A. tfidf_top_sentences: compute TF-IDF cosine between each sentence and full doc,
     keep top-N most representative sentences as "condensed query"
  B. structured_extract: title + abstract + first sentence of each section
  C. tfidf_keywords: select top-K highest-IDF words as query

Usage:
  python scripts/query_optimize.py \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --output data/queries_optimized.jsonl \
    --strategy tfidf_top_sentences \
    --max-sentences 20
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def split_sentences(text):
    """Split text into sentences."""
    return re.split(r'(?<=[.!?])\s+', text)


def tfidf_top_sentences(text, max_sentences=20, max_chars=3000):
    """
    Rank sentences by TF-IDF cosine similarity to the full document.
    Return top max_sentences concatenated, capped at max_chars.
    """
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text[:max_chars]

    # Build TF-IDF for sentences + full doc
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    all_texts = sentences + [text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Last row is the full document
    doc_vec = tfidf_matrix[-1].toarray().flatten()
    sent_vecs = tfidf_matrix[:-1].toarray()

    # Cosine similarity
    doc_norm = np.linalg.norm(doc_vec)
    if doc_norm == 0:
        return text[:max_chars]

    similarities = []
    for i, sent_vec in enumerate(sent_vecs):
        sent_norm = np.linalg.norm(sent_vec)
        if sent_norm == 0:
            similarities.append(0.0)
        else:
            sim = np.dot(sent_vec, doc_vec) / (sent_norm * doc_norm)
            similarities.append(sim)

    # Select top sentences
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    selected = [sentences[i] for i, _ in ranked[:max_sentences]]
    # Keep original order
    selected.sort(key=lambda s: sentences.index(s))

    result = ' '.join(selected)
    return result[:max_chars]


def structured_extract(text, max_chars=3000):
    """
    Extract title + abstract + first sentence of each section.
    Falls back to first max_chars if structure not detected.
    """
    # Try to find abstract
    abstract_match = re.search(
        r'(?i)abstract\b[:\s]*\n*(.*?)(?:\n\s*(?:\d+\.?\s*)?(?:introduction|chapter|section|\bI\b\.))',
        text, re.DOTALL
    )

    parts = []

    # First 2 lines as title area
    lines = text.split('\n')
    title_lines = [l for l in lines[:5] if l.strip() and len(l.strip()) > 10]
    if title_lines:
        parts.append(title_lines[0])

    # Abstract
    if abstract_match:
        parts.append('Abstract: ' + abstract_match.group(1).strip()[:1000])
    else:
        # Take first substantial paragraph as abstract
        paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        if paras:
            parts.append(paras[0][:1000])

    # First sentence of each section
    section_headers = re.split(
        r'\n\s*(?:\d+\.?\s*)?(?:introduction|related work|method|experiment|result|discussion|conclusion|references|acknowledgment)',
        text, flags=re.IGNORECASE
    )
    # Extract sentences around section starts
    section_pattern = re.finditer(
        r'(?:^|\n)\s*(?:\d+\.?\s*)?([A-Z][A-Za-z\s]{3,40})(?:\n|$|\s*\.)',
        text
    )
    for m in section_pattern:
        header = m.group(1).strip()
        if len(header) > 5 and header.lower() not in ('the', 'and', 'for', 'with', 'that', 'this'):
            parts.append(f'[Section: {header}]')

    result = ' '.join(parts)[:max_chars]
    if len(result) < 200:
        return text[:max_chars]
    return result


def tfidf_keywords(text, max_words=500, max_chars=3000):
    """
    Select top max_words highest-TF-IDF words as query.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_words)
    try:
        vectorizer.fit([text])
    except ValueError:
        return text[:max_chars]

    # Get words sorted by IDF
    idf = vectorizer.idf_
    vocab = vectorizer.get_feature_names_out()
    word_idf = list(zip(vocab, idf))
    word_idf.sort(key=lambda x: x[1], reverse=True)

    keywords = [w for w, _ in word_idf[:max_words]]
    return ' '.join(keywords)[:max_chars]


def optimize_query(text, strategy='tfidf_top_sentences', max_chars=3000, max_sentences=20):
    if strategy == 'tfidf_top_sentences':
        return tfidf_top_sentences(text, max_sentences=max_sentences, max_chars=max_chars)
    elif strategy == 'structured_extract':
        return structured_extract(text, max_chars=max_chars)
    elif strategy == 'tfidf_keywords':
        return tfidf_keywords(text, max_chars=max_chars)
    else:
        return text[:max_chars]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", default="tfidf_top_sentences",
                        choices=["tfidf_top_sentences", "structured_extract", "tfidf_keywords", "baseline"])
    parser.add_argument("--max-sentences", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    print(f"Strategy: {args.strategy}")
    print(f"Loading queries from {args.queries}...")

    with open(args.queries, encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            q = json.loads(line.strip())
            qid = q.get("qid") or q.get("query_id")
            text = q.get("query") or q.get("default_text") or ""

            if args.strategy == 'baseline':
                optimized = text[:args.max_chars]
            else:
                optimized = optimize_query(
                    text,
                    strategy=args.strategy,
                    max_sentences=args.max_sentences,
                    max_chars=args.max_chars,
                )

            entry = {"qid": qid, "query": optimized}
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if (i + 1) % 5000 == 0:
                print(f"  {i + 1} queries processed")

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
