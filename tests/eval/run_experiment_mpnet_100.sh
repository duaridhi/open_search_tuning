#!/usr/bin/env bash
# run_experiment_mpnet_100.sh
#
# Experiment: sentence-transformers/all-mpnet-base-v2 (768-d), first 100 contracts.
#
# Compare against run_experiment_minilm_50.sh to see:
#   - Quality difference between 384-d MiniLM and 768-d MPNet embeddings.
#   - Effect of a larger doc sample on recall coverage.
#
# Steps:
#   1. Create Qdrant collection with the right vector dimensions.
#   2. Ingest 100 contracts into that collection.
#   3. Build gold (chunk / contract / span projections) scoped to that collection.
#   4. Start the FastAPI server pointed at the collection.
#   5. Run retrieval eval; write results to tests/eval/runs/<collection>_<timestamp>/.
#
# Usage:
#   bash tests/eval/run_experiment_mpnet_100.sh
#
# Prerequisites:
#   - Qdrant reachable (set QDRANT_URL / CLUSTER_URL + QDRANT_API_KEY in cuad-demo-quadrant/.env)
#   - PDF_ROOT points to CUAD full_contract_pdf/ directory (or set via env)
#   - pip install -r requirements.txt already done

set -euo pipefail

# ── Experiment parameters ────────────────────────────────────────────────────
COLLECTION="cuad_sample_mpnet_100"
EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"
VECTOR_SIZE=768
DOC_OFFSET=0
DOC_COUNT=100
SERVER_PORT=8002
# SKIP_INGESTED_DOCS=1 (default): reuse the existing collection and skip any PDF
# whose title is already in Qdrant — safe to re-run after an interruption.
# Set SKIP_INGESTED_DOCS=0 to drop and recreate the collection from scratch.
SKIP_INGESTED_DOCS="${SKIP_INGESTED_DOCS:-1}"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INGEST_SCRIPT="$REPO_ROOT/cuad-demo-quadrant/upload_to_qdrant.py"
GOLD_SCRIPT="$REPO_ROOT/tests/eval/build_gold.py"
EVAL_SCRIPT="$REPO_ROOT/tests/eval/run_eval.py"
BASE_URL="http://localhost:$SERVER_PORT"
GOLD_DIR="$REPO_ROOT/tests/eval/gold/$COLLECTION"
RUN_OUT="$REPO_ROOT/tests/eval/runs/${COLLECTION}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "  Experiment         : $COLLECTION"
echo "  Model              : $EMBED_MODEL"
echo "  Vector dim         : $VECTOR_SIZE"
echo "  Docs               : $DOC_COUNT starting at offset $DOC_OFFSET"
echo "  SKIP_INGESTED_DOCS : $SKIP_INGESTED_DOCS"
echo "  Server             : $BASE_URL"
echo "  Gold dir           : $GOLD_DIR"
echo "  Output             : $RUN_OUT"
echo "============================================================"

# ── Step 1: Create collection (skipped when SKIP_INGESTED_DOCS=1) ────────────
echo ""
if [ "$SKIP_INGESTED_DOCS" = "1" ]; then
  echo ">>> [1/5] SKIP_INGESTED_DOCS=1 — reusing existing collection $COLLECTION"
else
  echo ">>> [1/5] Create collection $COLLECTION (${VECTOR_SIZE}-d cosine)"
  python3 - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/cuad-demo-quadrant")
from qdrant_cluster_connect import get_qdrant_client
from qdrant_client.models import Distance, VectorParams
client = get_qdrant_client()
existing = [c.name for c in client.get_collections().collections]
if "${COLLECTION}" in existing:
    client.delete_collection("${COLLECTION}")
    print("  Dropped existing collection ${COLLECTION}")
client.create_collection(
    "${COLLECTION}",
    vectors_config=VectorParams(size=${VECTOR_SIZE}, distance=Distance.COSINE),
)
print("  Collection ${COLLECTION} ready (${VECTOR_SIZE}-d cosine)")
PYEOF
fi

# ── Step 2: Ingest ───────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Ingest $DOC_COUNT contracts (offset=$DOC_OFFSET) → $COLLECTION"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
DOC_OFFSET="$DOC_OFFSET" \
DOC_COUNT="$DOC_COUNT" \
SKIP_INGESTED_DOCS="$SKIP_INGESTED_DOCS" \
  python3 "$INGEST_SCRIPT"

# ── Step 3: Build gold ───────────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Build gold for $COLLECTION"
QDRANT_COLLECTION="$COLLECTION" \
  python "$GOLD_SCRIPT" --collection "$COLLECTION" --out-dir "$GOLD_DIR"

# ── Step 4: Start server ─────────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Start API server on port $SERVER_PORT"
cd "$REPO_ROOT"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
  uvicorn app:app --host 0.0.0.0 --port "$SERVER_PORT" --log-level warning &
SERVER_PID=$!
trap 'echo "Stopping server (pid=$SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

echo -n "Waiting for server"
for i in $(seq 1 60); do
  sleep 2
  if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo " ready (${i}×2s elapsed)"
    break
  fi
  printf "."
  if [ "$i" -eq 60 ]; then
    echo ""
    echo "ERROR: server did not become ready within 120s"
    exit 1
  fi
done

# ── Step 5: Run eval ─────────────────────────────────────────────────────────
echo ""
echo ">>> [5/5] Run eval → $RUN_OUT"
python "$EVAL_SCRIPT" \
  --base-url "$BASE_URL" \
  --gold-dir "$GOLD_DIR" \
  --out "$RUN_OUT"

echo ""
echo "============================================================"
echo "  Done."
echo "  Summary : $RUN_OUT/summary.json"
echo "============================================================"
