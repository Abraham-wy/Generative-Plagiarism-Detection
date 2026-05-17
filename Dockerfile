FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    numpy \
    sentence-transformers

# Pre-download E5 model (cached in image — no network at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/e5-base-v2')"

# Copy entrypoint
WORKDIR /app
COPY scripts/submit_pan26.py /app/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/submit_pan26.py"]
