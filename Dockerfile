FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends lsb-release && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    torch --index-url https://download.pytorch.org/whl/cpu \
    "transformers==4.44.0" \
    "sentence-transformers==3.0.1"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"

WORKDIR /app
COPY scripts/submit_pan26.py /app/
COPY scripts/query_segment_retrieve.py /app/
COPY .git /app/.git
COPY .gitignore /app/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/submit_pan26.py"]
