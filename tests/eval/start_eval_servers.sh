#!/usr/bin/env bash
# start_eval_servers.sh
#
# Starts all 4 model servers used by compare_models.py.
#
#   voyage-law-2  → port 8006  (requires VOYAGE_API_KEY)
#   bge-large     → port 8013  (requires HF_TOKEN)
#   mpnet         → port 8012  (local SentenceTransformer)
#   minilm        → port 8011  (local SentenceTransformer)
#
# Usage:
#   source .env.dev && bash tests/eval/start_eval_servers.sh
#
#   # Or let the script load .env.dev itself:
#   bash tests/eval/start_eval_servers.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env.dev if not already sourced
ENV_FILE="$REPO_ROOT/.env.dev"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "Loaded $ENV_FILE"
fi

# ── Helper: start one server unless its port is already responding ─────────────

start_server() {
  local name="$1" port="$2"; shift 2
  if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
    echo "  $name  port=$port  already up — skipping"
    return
  fi
  echo "  Starting $name  port=$port  ..."
  env "$@" \
    uvicorn app:app --host 0.0.0.0 --port "$port" --log-level warning &
  echo "    pid=$!"
}

echo "Starting model servers..."
echo ""

start_server "voyage-law-2" 8006 \
  QDRANT_COLLECTION=cuad_voyage_law2_hybrid_10 \
  EMBED_MODEL=voyage-law-2 \
  EMBED_PROVIDER=voyageai \
  VECTOR_SIZE=1024 \
  SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
  ENABLE_HYBRID=1 \
  ENABLE_RERANKER=0 \
  SEARCH_TIMEOUT=180

start_server "voy-law2-200" 8007 \
  QDRANT_COLLECTION=cuad_voyage_law2_hybrid_200 \
  EMBED_MODEL=voyage-law-2 \
  EMBED_PROVIDER=voyageai \
  VECTOR_SIZE=1024 \
  SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
  ENABLE_HYBRID=1 \
  ENABLE_RERANKER=0 \
  SEARCH_TIMEOUT=180

start_server "bge-large" 8013 \
  QDRANT_COLLECTION=cuad_bgelarge_hybrid_50 \
  EMBED_MODEL=BAAI/bge-large-en-v1.5 \
  VECTOR_SIZE=1024 \
  SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
  ENABLE_HYBRID=1 \
  ENABLE_RERANKER=0 \
  EMBED_PROVIDER= \
  HF_PROVIDER=

start_server "mpnet" 8012 \
  QDRANT_COLLECTION=cuad_mpnet_hybrid_50 \
  EMBED_MODEL=sentence-transformers/all-mpnet-base-v2 \
  VECTOR_SIZE=768 \
  SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
  ENABLE_HYBRID=1 \
  ENABLE_RERANKER=0 \
  EMBED_PROVIDER= \
  HF_PROVIDER=

start_server "minilm" 8011 \
  QDRANT_COLLECTION=cuad_minilm_hybrid_50 \
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  VECTOR_SIZE=384 \
  SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
  ENABLE_HYBRID=1 \
  ENABLE_RERANKER=0 \
  EMBED_PROVIDER= \
  HF_PROVIDER=

# ── Wait for all 4 to be ready ─────────────────────────────────────────────────

echo ""
echo "Waiting for servers to be ready (up to 120s each) ..."
for port in 8006 8007 8013 8012 8011; do
  ready=0
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
      echo "  port $port  ready  (${i}×2s)"
      ready=1
      break
    fi
    sleep 2
  done
  [ "$ready" -eq 0 ] && echo "  port $port  TIMEOUT — check logs"
done

echo ""
echo "Run the comparison:"
echo "  python tests/eval/compare_models.py --doc 'DigitalCinema'"
echo "  python tests/eval/compare_models.py --doc 'DigitalCinema' --verbose"
echo "  python tests/eval/compare_models.py --doc 'DigitalCinema' --multi-query"
echo "  python tests/eval/compare_models.py --doc 'DigitalCinema' --multi-query --verbose"
echo "  python tests/eval/compare_models.py --list-docs"
