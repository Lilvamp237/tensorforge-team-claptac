# One image for all three services; docker-compose.yml picks the command (API, Streamlit UI or MCP server).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY data/kb/ data/kb/
COPY data/synthetic/ data/synthetic/
COPY models/ models/
RUN mkdir -p logs data/processed data/raw

EXPOSE 8000 8501 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Build any missing models (first start only), then serve.
CMD ["sh", "-c", "python scripts/bootstrap.py && exec uvicorn safepassage.api:app --host 0.0.0.0 --port 8000"]
