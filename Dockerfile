FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PAN_MODEL=/models/all-MiniLM-L6-v2 \
    PAN_THRESHOLD=0.80 \
    PAN_OUTPUT_FORMAT=xml

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build-time download only. Runtime can run offline because the model is baked
# into the image and selected through PAN_MODEL.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2").save("/models/all-MiniLM-L6-v2")
PY

COPY main.py ./main.py

ENTRYPOINT ["python", "/app/main.py"]
CMD ["/input", "/output"]
