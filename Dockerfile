FROM python:3.11-slim

# ── system dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
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
COPY src/       ./src/
COPY predict.py ./predict.py
COPY train.py   ./train.py

# ── TIRA / PAN entrypoint ────────────────────────────────────────────────────
# TIRA mounts:
#   /input  → directory containing the test JSONL (or XML) corpus
#   /output → directory where predictions should be written
#
# Set DETECTOR_MODE to "features", "finetune", "zeroshot", or "roberta"
# Set MODEL_PATH  to the path of a pre-trained model file/directory
# Set PPL_THRESHOLD for zeroshot mode (default 50)
ENV DETECTOR_MODE=zeroshot \
    MODEL_PATH="" \
    PPL_THRESHOLD=50.0 \
    INPUT_FILE=/input/test.jsonl \
    OUTPUT_FILE=/output/predictions.jsonl

CMD python predict.py \
    --input  "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --mode   "$DETECTOR_MODE" \
    --model  "$MODEL_PATH" \
    --threshold "$PPL_THRESHOLD"
