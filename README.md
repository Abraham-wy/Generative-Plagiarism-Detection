# Query-Chunk Provenance Modeling for Generative Plagiarism Detection

Source retrieval system for the [PAN@CLEF 2026 Generative Plagiarism Detection](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html) task. Official TIRA submission of **Team Fosuai** (software: `oscillating-list`).

## Paper

Yi Wang, Zhongyuan Han, Chang Liu, Haojie Cao. *Team Fosuai at PAN 2026: Source Retrieval via Query-Chunk Provenance Modeling for Generative Plagiarism Detection.* CLEF 2026 Working Notes.

## Method

Given a suspicious document potentially synthesized from multiple sources, we decompose it into paragraph-level query chunks, perform BM25 retrieval independently per chunk, and aggregate candidates via coverage-weighted voting. Documents matched by multiple independent chunks receive a coverage bonus, strengthening signals from sources with genuine multi-paragraph provenance.

**Pipeline:** Suspicious document → paragraph-level query chunks → independent BM25 retrieval → segment-level candidate lists → coverage-weighted final ranking.

**Key features:**
- Only requires `numpy` and Python standard library — no GPU or pre-trained language models
- Traditional inverted index with BM25 ranking
- Paragraph-boundary-aware segmentation (min 800 chars, max 3,000 chars per chunk)
- Coverage-weighted voting: `Score(d) = Σ BM25(d|c_i) × (1 + count(d) / n)`

## Official Results (PAN26 Main Test)

| System | Recall@10 | nDCG@10 | RR |
|--------|-----------|---------|-----|
| BM25 baseline | 0.5767 | 0.5532 | 0.8318 |
| **fosuai (ours)** | 0.6300 | 0.6263 | 0.9350 |
| JLP (best) | 0.8542 | 0.7994 | 0.9760 |

Our system ranked 3rd by Reciprocal Rank (RR=0.9350).

## Project Structure

```
.
├── Dockerfile              # TIRA submission Docker image
├── scripts/
│   ├── submit_pan26.py     # TIRA entry point (main method)
│   ├── evaluate.py         # IR evaluation (nDCG, Recall, MRR, MAP)
│   ├── bm25_index.py       # BM25 inverted index construction
│   ├── baseline_bm25.py    # Standard BM25 baseline
│   ├── bm25_top100.py      # BM25 top-100 retrieval
│   ├── query_segment_retrieve.py  # Query-chunk retrieval pipeline
│   ├── run_segmentation.py        # Batch segmentation experiments
│   ├── segmentation_eval.py       # Segmentation strategy evaluation
│   ├── seg_compare.py             # Segmentation comparison
│   ├── dense_encode.py     # E5 dense embedding
│   ├── dense_retrieve.py   # Dense retrieval
│   ├── dense_chunk.py      # E5 chunking (256-token windows)
│   ├── query_decompose.py  # Query decomposition baseline
│   ├── export_bm25_top100.py      # Export retrieval results
│   ├── convert_pan25.py    # PAN25→PAN26 format conversion
│   ├── pan26_e2e.py        # End-to-end pipeline
│   └── ...                 # Additional experimental scripts
└── docs/                   # Paper source files
```

## Quick Start

### Requirements

- Python 3.10+
- numpy

```bash
pip install numpy
```

### TIRA Submission (Docker)

```bash
docker build -t pan26-submission .
docker run --rm \
  -v /path/to/dataset:/data/pan26/test-dataset \
  -v /path/to/output:/tmp/pan26_output \
  pan26-submission \
  --input /data/pan26/test-dataset \
  --output /tmp/pan26_output
```

### Local Evaluation

```bash
# Standard BM25 baseline
python scripts/baseline_bm25.py --corpus corpus.jsonl --queries queries.jsonl --output run.txt

# Query-chunk provenance modeling (our method)
python scripts/query_segment_retrieve.py --corpus corpus.jsonl --queries queries.jsonl --output run.txt

# Evaluate results
python scripts/evaluate.py --run run.txt --qrels qrels.txt
```

### Run Segmentation Experiments

```bash
# Batch segmentation
python scripts/run_segmentation.py --queries queries.jsonl --output results/

# Compare segmentation strategies
python scripts/seg_compare.py --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.txt

# Detailed segmentation evaluation
python scripts/segmentation_eval.py --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.txt
```

## Data

The PAN26 task uses 60,592 source documents from ClueWeb09. Development data is derived from the PAN 2025 generative plagiarism detection dataset.

- Task overview: [PAN26 Generative Plagiarism Detection](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html)
- TIRA submission platform: [TIRA](https://www.tira.io/)

## Citation

```bibtex
@inproceedings{wang:2026,
  author    = {Yi Wang and Zhongyuan Han and Chang Liu and Haojie Cao},
  title     = {Team Fosuai at PAN 2026: Source Retrieval via Query-Chunk
               Provenance Modeling for Generative Plagiarism Detection},
  booktitle = {CLEF 2026 Working Notes},
  year      = {2026},
}
```

## License

This project is licensed under CC BY 4.0.
