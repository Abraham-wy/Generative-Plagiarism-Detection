FROM python:3.11-slim

# ── system dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python - <<'EOF'
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)
EOF

# ── Application code ─────────────────────────────────────────────────────────
COPY src/        ./src/
COPY predict.py  ./predict.py
COPY train.py    ./train.py
COPY retrieve.py ./retrieve.py

# ── TIRA / PAN entrypoint ─────────────────────────────────────────────────────
# PAN 2026 task: retrieve source documents for AI-generated suspicious documents.
# Output: run.txt.gz in TREC run format (at most 1000 results per query).
#
# Environment variables:
#   RETRIEVAL_DATASET  = ir_datasets ID or path to local data directory (default: /input)
#   RETRIEVAL_INDEX    = directory where BM25 index is stored/built    (default: /tmp/indexes)
#   RETRIEVAL_TAG      = system tag written into the TREC run file      (default: pan26-retrieval)
#   RETRIEVAL_RERANK   = "true" to enable dense re-ranking (default),
#                        "false" for pure BM25+RRF mode (faster)
ENV RETRIEVAL_DATASET=/input \
    RETRIEVAL_INDEX=/tmp/indexes \
    RETRIEVAL_TAG=pan26-retrieval \
    RETRIEVAL_RERANK=true

CMD if [ "$RETRIEVAL_RERANK" = "true" ]; then RERANK_FLAG="--rerank"; else RERANK_FLAG="--no-rerank"; fi; \
    python retrieve.py \
        --dataset "$RETRIEVAL_DATASET" \
        --output  /output \
        --index   "$RETRIEVAL_INDEX" \
        --tag     "$RETRIEVAL_TAG" \
        $RERANK_FLAG

