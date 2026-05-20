FROM python:3.11-slim

RUN pip install --no-cache-dir numpy

WORKDIR /app
COPY scripts/submit_pan26.py /app/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/submit_pan26.py"]
