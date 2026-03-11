#!/bin/bash
set -e

echo "[STARTUP] Downloading NLTK data..."
python -m nltk.downloader punkt -d /home/user/nltk_data

echo "[STARTUP] Starting Embedding Service on port 8001..."
cd /app/cuad-demo-quadrant/embeddings
export PYTHONPATH=/app/cuad-demo-quadrant:$PYTHONPATH
python -m uvicorn embedding_service:app --host 0.0.0.0 --port 8001 &
EMBEDDING_PID=$!

echo "[STARTUP] Waiting for Embedding Service to be ready..."
sleep 10

echo "[STARTUP] Starting Main API on port 7860..."
cd /app
exec python -m uvicorn app:app --host 0.0.0.0 --port 7860

# Cleanup on exit
trap "kill $EMBEDDING_PID" EXIT
