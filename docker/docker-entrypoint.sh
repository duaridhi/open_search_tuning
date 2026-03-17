#!/bin/bash
set -e

# echo "[STARTUP] Starting Embedding Service on port 8001..."
# echo "[STARTUP] Waiting for Embedding Service to be ready..."
echo "[STARTUP] Starting Main API on port 7860..."
if [ -z "$HF_TOKEN" ]; then
	echo "[ERROR] HF_TOKEN environment variable must be set for HuggingFace Inference API."
	exit 1
fi

echo "[STARTUP] Starting HuggingFace Embedding/Highlighting Service on port 8001..."
cd /app/cuad-demo-quadrant/embeddings
export PYTHONPATH=/app/cuad-demo-quadrant:$PYTHONPATH
python -m uvicorn embedding_service_hf:app --host 0.0.0.0 --port 8001 &
EMBEDDING_PID=$!

echo "[STARTUP] Waiting for Embedding Service to be ready..."
sleep 10

echo "[STARTUP] Starting Main API (using qdrant_search_hf) on port 7860..."
cd /app
# Optionally, update app.py to import from qdrant_search_hf instead of qdrant_search
exec python -m uvicorn app:app --host 0.0.0.0 --port 7860

# Cleanup on exit
trap "kill $EMBEDDING_PID" EXIT
