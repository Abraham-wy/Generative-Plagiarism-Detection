FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.retrieval-lite.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements.retrieval-lite.txt

RUN python -m nltk.downloader punkt punkt_tab stopwords

# Pre-initialize PyTerrier to cache Maven/Terrier JARs at build time (runtime is offline)
RUN python -c "import pyterrier as pt; pt.init()"

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("multi-qa-mpnet-base-dot-v1").save("/models/multi-qa-mpnet")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").save("/models/cross-encoder-msmarco")
PY

ENV PAN_CROSS_ENCODER=/models/cross-encoder-msmarco
ENV PAN_MODEL=/models/multi-qa-mpnet

COPY retrieve.py ./retrieve.py

ENTRYPOINT ["python", "/app/retrieve.py"]
CMD ["--dataset", "$inputDataset", "--index", "/tmp/indexes", "--output", "$outputDir"]
