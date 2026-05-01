# PAN Plagiarism Detection Offline Baseline

This repository now provides a runnable baseline for the PAN generated plagiarism
text-alignment task: given document pairs, detect reused passages between a
suspicious document and its source document.

The system is fully local at runtime:

- no PyTerrier
- no Java
- no external API
- sentence-transformers embeddings + NumPy cosine similarity
- Docker entrypoint: `python main.py /input /output`

## Task Format

Expected input directory:

```text
/input/
├── pairs
├── susp/
│   └── suspicious-document001.txt
└── src/
    └── source-document001.txt
```

The `pairs` file contains one pair per line:

```text
suspicious-document001.txt source-document001.txt
suspicious-document002.txt source-document015.txt
```

For each pair, the system writes one XML file to `/output`:

```xml
<?xml version='1.0' encoding='utf-8'?>
<document reference="suspicious-document001.txt">
  <feature
    name="detected-plagiarism"
    this_offset="123"
    this_length="456"
    source_reference="source-document001.txt"
    source_offset="789"
    source_length="456" />
</document>
```

This follows the PAN text-alignment format described on the PAN generated
plagiarism detection task page: output features should be named
`detected-plagiarism` and include suspicious/source offsets and lengths.

## How It Works

1. Read each pair from `pairs`.
2. Load the suspicious and source documents.
3. Split each document into sentence-like spans while preserving character offsets.
4. Build overlapping windows from the sentence spans.
5. Encode windows with `sentence-transformers` using `all-MiniLM-L6-v2` by default.
6. Compute cosine similarity between suspicious and source windows.
7. Emit detections where similarity is above the threshold.
8. Merge nearby detections into longer passages.
9. Write PAN-style XML files.

The approach is inspired by indexless raw-text retrieval systems such as
Sirchmunk: keep the comparison local to the input pair, chunk raw text, and
match semantically without a persistent search index.

## Local Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python main.py /path/to/dataset /path/to/output
```

Example with custom threshold:

```bash
python main.py /path/to/dataset /path/to/output --threshold 0.78
```

Optional JSONL output for debugging:

```bash
python main.py /path/to/dataset /path/to/output --format jsonl
```

## Docker Run

Build:

```bash
docker build -t pan-plagiarism-baseline .
```

Run:

```bash
docker run --rm \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline
```

The Docker image downloads and stores `all-MiniLM-L6-v2` at build time under
`/models/all-MiniLM-L6-v2`, so runtime execution can be offline.

## Parameters

CLI flags and matching environment variables:

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--model` | `PAN_MODEL` | `all-MiniLM-L6-v2` locally, `/models/all-MiniLM-L6-v2` in Docker | sentence-transformers model |
| `--threshold` | `PAN_THRESHOLD` | `0.80` | cosine threshold for plagiarism windows |
| `--window-chars` | `PAN_WINDOW_CHARS` | `550` | maximum chars per comparison window |
| `--step-sentences` | `PAN_STEP_SENTENCES` | `2` | sentence stride for overlapping windows |
| `--min-chars` | `PAN_MIN_CHARS` | `80` | minimum window length |
| `--batch-size` | `PAN_BATCH_SIZE` | `32` | embedding batch size |
| `--top-k` | `PAN_TOP_K` | `2` | source matches kept per suspicious window |
| `--merge-gap` | `PAN_MERGE_GAP` | `120` | max char gap for merging adjacent hits |
| `--max-detections` | `PAN_MAX_DETECTIONS` | `200` | output cap per pair |
| `--format` | `PAN_OUTPUT_FORMAT` | `xml` | `xml` or `jsonl` |

Recommended first experiments:

```bash
python main.py /input /output --threshold 0.80 --window-chars 550
python main.py /input /output --threshold 0.75 --window-chars 700
python main.py /input /output --threshold 0.82 --step-sentences 1
```

Lower thresholds improve recall but may add false positives; larger windows
often help paraphrased plagiarism but make offsets coarser.

## Files

```text
main.py              # offline PAN text-alignment baseline
requirements.txt     # minimal runtime dependencies
Dockerfile           # Python 3.10 offline-runtime container
Dockerfile.lite      # same lightweight baseline without extra legacy tooling
docs/BEGINNER_PROJECT_GUIDE.md # step-by-step guide for first-time participants
```

Legacy exploratory scripts from earlier retrieval experiments may still exist in
the repository, but the baseline required here is `main.py`.

For a zero-background walkthrough covering data layout, local runs, Docker,
TIRA submission, scoring, and tuning, read
[`docs/BEGINNER_PROJECT_GUIDE.md`](docs/BEGINNER_PROJECT_GUIDE.md).
