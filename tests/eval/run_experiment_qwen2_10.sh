#!/usr/bin/env bash
# run_experiment_qwen2_10.sh
#
# Experiment: Alibaba-NLP/gte-Qwen2-1.5B-instruct (1536-d), first 10 contracts.
#
# Compare against run_experiment_minilm_50.sh to see:
#   - Quality difference between 384-d MiniLM and 1536-d Qwen2-instruct embeddings.
#   - Whether a much larger embedding model improves span-hit and contract recall
#     even on a tiny sample.
#
# Model notes:
#   - gte-Qwen2-1.5B-instruct is an instruction-tuned embedding model.
#   - Document chunks are encoded WITHOUT any instruction prefix (raw passage).
#   - Queries should be prefixed with a task instruction at search time:
#       "Instruct: Retrieve relevant contract clauses\nQuery: <query>"
#     This prefix is NOT applied here (ingest only). Wire it into
#     qdrant_search_hf.py (EMBED_QUERY_PREFIX env var or equivalent) before
#     running comparative evals.
#   - First run downloads ~3 GB from HuggingFace. Cached in ~/.cache/huggingface/.
#   - CPU inference: ~3–6 GB RAM per batch. Expect ~1–5 min for 10 docs on CPU.
#
# Steps:
#   1. Create Qdrant collection (1536-d cosine).
#   2. Ingest 10 contracts using gte-Qwen2-1.5B-instruct embeddings.
#   3. Build gold (chunk / contract / span projections) for this collection.
#   4. Start the FastAPI server pointed at the collection.
#   5. Run retrieval eval; write results to tests/eval/runs/<collection>_<timestamp>/.
#
# Usage:
#   bash tests/eval/run_experiment_qwen2_10.sh
#
# Prerequisites:
#   - Qdrant reachable (set QDRANT_URL / CLUSTER_URL + QDRANT_API_KEY in cuad-demo-quadrant/.env)
#   - PDF_ROOT points to CUAD full_contract_pdf/ directory (or set via env)
#   - pip install -r requirements.txt already done (includes sentence-transformers)

set -euo pipefail

# ── Experiment parameters ────────────────────────────────────────────────────
COLLECTION="cuad_qwen2_10"
EMBED_MODEL="BAAI/bge-large-en-v1.5"
#EMBED_MODEL="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
#EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
HF_PROVIDER="hf-inference"
VECTOR_SIZE=1024
DOC_COUNT=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SERVER_PORT=8004
# SKIP_INGESTED_DOCS=1 (default): reuse the existing collection and skip any PDF
# whose title is already in Qdrant — safe to re-run after an interruption.
# Set SKIP_INGESTED_DOCS=0 to drop and recreate the collection from scratch.
SKIP_INGESTED_DOCS=1
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
echo "  Model              : $EMBED_MODEL"
echo "  Vector dim         : $VECTOR_SIZE"
echo "  Docs               : $DOC_COUNT"
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
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
client = get_qdrant_client()
existing = [c.name for c in client.get_collections().collections]
if "${COLLECTION}" in existing:
    client.delete_collection("${COLLECTION}")
    print("  Dropped existing collection ${COLLECTION}")
client.create_collection(
    "${COLLECTION}",
    vectors_config=VectorParams(size=${VECTOR_SIZE}, distance=Distance.COSINE),
)
try:
    client.create_payload_index("${COLLECTION}", "title", PayloadSchemaType.KEYWORD)
except Exception:
    pass
print("  Collection ${COLLECTION} ready (${VECTOR_SIZE}-d cosine)")
PYEOF
fi

# ── Step 2: Ingest documents ─────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Ingest $DOC_COUNT contracts with $EMBED_MODEL"
# Uses upload_to_qdrant_hf.py — embeddings generated via HF Inference API,
# no local model RAM. Documents encoded WITHOUT an instruction prefix (query-side only).
# VECTOR_SIZE is passed explicitly so collection creation doesn't need a probe call.
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
HF_PROVIDER="$HF_PROVIDER" \
VECTOR_SIZE="$VECTOR_SIZE" \
DOC_COUNT="$DOC_COUNT" \
CHUNK_SIZE="$CHUNK_SIZE" \
CHUNK_OVERLAP="$CHUNK_OVERLAP" \
ENCODE_BATCH_SIZE=8 \
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
python3 "$EVAL_SCRIPT" \
  --base-url "$BASE_URL" \
  --gold-dir "$GOLD_DIR" \
  --out "$RUN_OUT"

echo ""
echo "============================================================"
echo "  Done."
echo "  Summary  : $RUN_OUT/summary.json"
echo "  Gold     : $GOLD_DIR"
echo ""
echo "  To compare against MiniLM baseline:"
echo "    python3 -c \""
echo "      import json"
echo "      a = json.load(open('$RUN_OUT/summary.json'))"
echo "      b = json.load(open('tests/eval/runs/<minilm-run>/summary.json'))"
echo "      # compare a vs b"
echo "    \""
echo "============================================================"
