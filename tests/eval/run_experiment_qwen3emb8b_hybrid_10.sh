#!/usr/bin/env bash
# run_experiment_qwen3emb8b_hybrid_10.sh
#
# Experiment: Qwen/Qwen3-Embedding-8B (4096-d dense) + BM42 sparse,
# hybrid search via Qdrant RRF fusion, first 10 contracts.
#
# Embeddings generated via HF Inference API (Scaleway provider — status=live).
# No local model RAM needed.
#
# Compare against:
#   run_experiment_bgelarge_hybrid_50.sh  (BAAI/bge-large-en-v1.5, 1024-d)
#   run_experiment_minilm_hybrid_50.sh   (all-MiniLM-L6-v2, 384-d)
#
# Usage:
#   bash tests/eval/run_experiment_qwen3emb8b_hybrid_10.sh

set -euo pipefail

# ── Experiment parameters ────────────────────────────────────────────────────
COLLECTION="cuad_qwen3emb8b_hybrid_10"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
VECTOR_SIZE=4096
SPARSE_MODEL="Qdrant/bm42-all-minilm-l6-v2-attentions"
HF_PROVIDER="scaleway"
DOC_OFFSET=0
DOC_COUNT=10
CHUNK_SIZE=500
CHUNK_OVERLAP=50
ENCODE_BATCH_SIZE=2
SERVER_PORT=8005
SKIP_INGESTED_DOCS="${SKIP_INGESTED_DOCS:-0}"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INGEST_SCRIPT="$REPO_ROOT/cuad-demo-quadrant/upload_to_qdrant_hf.py"
GOLD_SCRIPT="$REPO_ROOT/tests/eval/build_gold.py"
EVAL_SCRIPT="$REPO_ROOT/tests/eval/run_eval.py"
BASE_URL="http://localhost:$SERVER_PORT"
GOLD_DIR="$REPO_ROOT/tests/eval/gold/$COLLECTION"
MODEL_SLUG="$(echo "$EMBED_MODEL" | tr '/: ' '_')"
RUN_OUT="$REPO_ROOT/tests/eval/runs/${COLLECTION}_${MODEL_SLUG}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "  Experiment         : $COLLECTION"
echo "  Dense model        : $EMBED_MODEL ($VECTOR_SIZE-d)"
echo "  Sparse model       : $SPARSE_MODEL"
echo "  HF provider        : $HF_PROVIDER"
echo "  Docs               : $DOC_COUNT starting at offset $DOC_OFFSET"
echo "  SKIP_INGESTED_DOCS : $SKIP_INGESTED_DOCS"
echo "  Server             : $BASE_URL"
echo "  Gold dir           : $GOLD_DIR"
echo "  Output             : $RUN_OUT"
echo "============================================================"

# ── Step 1: Create collection ────────────────────────────────────────────────
echo ""
if [ "$SKIP_INGESTED_DOCS" = "1" ]; then
  echo ">>> [1/5] SKIP_INGESTED_DOCS=1 — reusing existing collection $COLLECTION"
else
  echo ">>> [1/5] Create collection $COLLECTION (${VECTOR_SIZE}-d dense + sparse)"
  python3 - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/cuad-demo-quadrant")
from qdrant_cluster_connect import get_qdrant_client
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PayloadSchemaType
client = get_qdrant_client()
existing = [c.name for c in client.get_collections().collections]
if "${COLLECTION}" in existing:
    client.delete_collection("${COLLECTION}")
    print("  Dropped existing collection ${COLLECTION}")
client.create_collection(
    "${COLLECTION}",
    vectors_config=VectorParams(size=${VECTOR_SIZE}, distance=Distance.COSINE),
    sparse_vectors_config={"sparse": SparseVectorParams()},
)
try:
    client.create_payload_index("${COLLECTION}", "title", PayloadSchemaType.KEYWORD)
except Exception:
    pass
print("  Collection ${COLLECTION} ready (${VECTOR_SIZE}-d dense + sparse)")
PYEOF
fi

# ── Step 2: Ingest ───────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Ingest $DOC_COUNT contracts with $EMBED_MODEL + $SPARSE_MODEL"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
VECTOR_SIZE="$VECTOR_SIZE" \
HF_PROVIDER="$HF_PROVIDER" \
SPARSE_MODEL="$SPARSE_MODEL" \
ENABLE_HYBRID="1" \
DOC_OFFSET="$DOC_OFFSET" \
DOC_COUNT="$DOC_COUNT" \
CHUNK_SIZE="$CHUNK_SIZE" \
CHUNK_OVERLAP="$CHUNK_OVERLAP" \
ENCODE_BATCH_SIZE="$ENCODE_BATCH_SIZE" \
SKIP_INGESTED_DOCS="$SKIP_INGESTED_DOCS" \
  python3 "$INGEST_SCRIPT"

# ── Step 3: Build gold ───────────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Build gold for $COLLECTION"
QDRANT_COLLECTION="$COLLECTION" \
  python3 "$GOLD_SCRIPT" --collection "$COLLECTION" --out-dir "$GOLD_DIR"

# ── Step 4: Start server ─────────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Start API server on port $SERVER_PORT"
cd "$REPO_ROOT"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
HF_PROVIDER="$HF_PROVIDER" \
SPARSE_MODEL="$SPARSE_MODEL" \
ENABLE_HYBRID="1" \
  uvicorn app:app --host 0.0.0.0 --port "$SERVER_PORT" --log-level warning &
SERVER_PID=$!
trap 'echo "Stopping server (pid=$SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

echo -n "Waiting for server"
for i in $(seq 1 90); do
  sleep 2
  if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo " ready (${i}×2s elapsed)"
    break
  fi
  printf "."
  if [ "$i" -eq 90 ]; then
    echo ""
    echo "ERROR: server did not become ready within 180s"
    exit 1
  fi
done

# ── Step 5: Run eval ─────────────────────────────────────────────────────────
echo ""
echo ">>> [5/5] Run eval → $RUN_OUT"
python3 "$EVAL_SCRIPT" \
  --base-url "$BASE_URL" \
  --gold-dir "$GOLD_DIR" \
  --strategy hybrid_search \
  --out "$RUN_OUT"

echo ""
echo "============================================================"
echo "  Done."
echo "  Summary : $RUN_OUT/summary.json"
echo "============================================================"
