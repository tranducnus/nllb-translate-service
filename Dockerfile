FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Download model at build time so container starts instantly.
# Using distilled 600M with int8 quantization — ~350MB on disk, ~1GB RAM, fast on CPU.
# JustFrederik/nllb-200-distilled-600M-ct2-int8 is a pre-converted CTranslate2 int8 checkpoint.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('JustFrederik/nllb-200-distilled-600M-ct2-int8', local_dir='/app/model')" && \
    ls -la /app/model/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
