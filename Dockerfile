FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends lsb-release && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy

WORKDIR /app
COPY scripts/submit_pan26.py /app/
COPY .git /app/.git

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/submit_pan26.py"]
