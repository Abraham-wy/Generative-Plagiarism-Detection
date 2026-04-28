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

# Make retrieve.py executable (for TIRA code-submission style invocation)
RUN chmod +x retrieve.py

# ── TIRA / PAN entrypoint ─────────────────────────────────────────────────────
# ---- Sub-task A: Classification (human vs AI) ----
# Set TASK=classify and configure as before:
#   DETECTOR_MODE = features | finetune | zeroshot | roberta
#   MODEL_PATH    = path to saved model file / directory
#   PPL_THRESHOLD = perplexity threshold (zeroshot mode only)
#
# ---- Sub-task B: Retrieval (find source documents) ----
# Set TASK=retrieve and configure:
#   RETRIEVAL_DATASET  = ir_datasets ID or path to local data dir
#   RETRIEVAL_INDEX    = directory where BM25 index is stored/built
#   RETRIEVAL_TAG      = system tag in the TREC run file
#   RETRIEVAL_RERANK   = "true" (default) or "false" for dense re-ranking
ENV TASK=classify \
    DETECTOR_MODE=zeroshot \
    MODEL_PATH="" \
    PPL_THRESHOLD=50.0 \
    INPUT_FILE=/input/test.jsonl \
    OUTPUT_FILE=/output/predictions.jsonl \
    RETRIEVAL_DATASET=/input \
    RETRIEVAL_INDEX=/tmp/indexes \
    RETRIEVAL_TAG=pan26-retrieval \
    RETRIEVAL_RERANK=true

CMD if [ "$TASK" = "retrieve" ]; then \
        RERANK_FLAG="--rerank"; \
        if [ "$RETRIEVAL_RERANK" = "false" ]; then RERANK_FLAG="--no-rerank"; fi; \
        python retrieve.py \
            --dataset  "$RETRIEVAL_DATASET" \
            --output   /output \
            --index    "$RETRIEVAL_INDEX" \
            --tag      "$RETRIEVAL_TAG" \
            $RERANK_FLAG; \
    else \
        python predict.py \
            --input    "$INPUT_FILE" \
            --output   "$OUTPUT_FILE" \
            --mode     "$DETECTOR_MODE" \
            --model    "$MODEL_PATH" \
            --threshold "$PPL_THRESHOLD"; \
    fi
