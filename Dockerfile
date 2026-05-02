FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.retrieval-lite.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements.retrieval-lite.txt

RUN python -m nltk.downloader punkt punkt_tab stopwords

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2").save("/models/all-MiniLM-L6-v2")
PY

COPY retrieve.py ./retrieve.py

ENTRYPOINT ["python", "/app/retrieve.py"]
CMD ["--dataset", "$inputDataset", "--index", "/tmp/indexes", "--output", "$outputDir"]
