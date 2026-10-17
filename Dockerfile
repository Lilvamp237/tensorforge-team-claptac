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
COPY models/ models/
RUN mkdir -p logs data/processed data/raw

EXPOSE 8000 8501 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "safepassage.api:app", "--host", "0.0.0.0", "--port", "8000"]
