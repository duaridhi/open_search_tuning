#!/usr/bin/env bash
# run_experiment_voyage_law2_hybrid_200.sh
#
# Experiment: voyage-law-2 (1024-d) + BM42 sparse, hybrid search,
# 200-char chunks (vs 500-char in cuad_voyage_law2_hybrid_10).
#
# Purpose: Measure the effect of smaller chunk size on retrieval metrics.
# Compare summary.json against:
#   cuad_voyage_law2_hybrid_10  (same model, 500-char chunks)
#   cuad_bgelarge_hybrid_50     (different model, 500-char chunks)
#   cuad_mpnet_hybrid_50        (different model, 500-char chunks)
#   cuad_minilm_hybrid_50       (different model, 500-char chunks)
#
# NOTE: VoyageAI free tier is 3 req/min.  With ENCODE_BATCH_SIZE=32 and
# ENCODE_SLEEP_S=21 the ingest stays within that limit.  Ingesting 50
# contracts (~18 000 chunks) takes ~3 hours.  Run overnight or in tmux.
#
# DigitalCinema (offset 2) is already in cuad_voyage_law2_hybrid_200;
# SKIP_INGESTED_DOCS=1 skips it automatically.
#
# Usage:
#   bash tests/eval/run_experiment_voyage_law2_hybrid_200.sh

set -euo pipefail

# ── Load .env (picks up VOYAGE_API_KEY, HF_TOKEN, QDRANT_*, etc.) ────────────
REPO_ROOT_FOR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT_FOR_ENV/cuad-demo-quadrant/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "Loaded env from $ENV_FILE"
fi

# Also load .env.dev (VOYAGE_API_KEY lives there in dev)
DEV_ENV="$REPO_ROOT_FOR_ENV/.env.dev"
if [ -f "$DEV_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEV_ENV"
  set +a
fi

# ── Experiment parameters ────────────────────────────────────────────────────
COLLECTION="cuad_voyage_law2_hybrid_200"
EMBED_MODEL="voyage-law-2"
EMBED_PROVIDER="voyageai"
VECTOR_SIZE=1024
SPARSE_MODEL="Qdrant/bm42-all-minilm-l6-v2-attentions"
DOC_OFFSET=0
DOC_COUNT=50
CHUNK_SIZE=200
CHUNK_OVERLAP=50
ENCODE_BATCH_SIZE=32
ENCODE_SLEEP_S=21
SERVER_PORT=8007
SKIP_INGESTED_DOCS=1

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
echo "  Dense model        : $EMBED_MODEL ($VECTOR_SIZE-d, legal-domain)"
echo "  Sparse model       : $SPARSE_MODEL"
echo "  Embed provider     : $EMBED_PROVIDER"
echo "  Chunk size / overlap: $CHUNK_SIZE / $CHUNK_OVERLAP"
echo "  Docs               : $DOC_COUNT starting at offset $DOC_OFFSET"
echo "  SKIP_INGESTED_DOCS : $SKIP_INGESTED_DOCS (DigitalCinema already ingested)"
echo "  Server             : $BASE_URL"
echo "  Gold dir           : $GOLD_DIR"
echo "  Output             : $RUN_OUT"
echo "  Est. ingest time   : ~3 h  (VoyageAI 3 req/min, ~18k chunks)"
echo "============================================================"

# ── Step 1: Collection already exists — skip creation ────────────────────────
echo ""
echo ">>> [1/5] Collection $COLLECTION already exists — skipping creation"

# ── Step 2: Ingest remaining docs ────────────────────────────────────────────
echo ""
echo ">>> [2/5] Ingest $DOC_COUNT contracts with $EMBED_MODEL + $SPARSE_MODEL"
echo "          (SKIP_INGESTED_DOCS=1 skips DigitalCinema already present)"
cd "$REPO_ROOT/cuad-demo-quadrant"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
EMBED_PROVIDER="$EMBED_PROVIDER" \
VECTOR_SIZE="$VECTOR_SIZE" \
SPARSE_MODEL="$SPARSE_MODEL" \
ENABLE_HYBRID="1" \
DOC_OFFSET="$DOC_OFFSET" \
DOC_COUNT="$DOC_COUNT" \
CHUNK_SIZE="$CHUNK_SIZE" \
CHUNK_OVERLAP="$CHUNK_OVERLAP" \
ENCODE_BATCH_SIZE="$ENCODE_BATCH_SIZE" \
ENCODE_SLEEP_S="$ENCODE_SLEEP_S" \
SKIP_INGESTED_DOCS="$SKIP_INGESTED_DOCS" \
  python3 "$INGEST_SCRIPT"
cd "$REPO_ROOT"

# ── Step 3: Build gold ───────────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Build gold for $COLLECTION → $GOLD_DIR"
QDRANT_COLLECTION="$COLLECTION" \
  python3 "$GOLD_SCRIPT" --collection "$COLLECTION" --out-dir "$GOLD_DIR"

# ── Step 4: Start server ─────────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Start API server on port $SERVER_PORT"
# Kill any existing process on this port
if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
  echo "  Port $SERVER_PORT already responding — stopping old process"
  fuser -k "${SERVER_PORT}/tcp" 2>/dev/null || true
  sleep 2
fi
cd "$REPO_ROOT"
QDRANT_COLLECTION="$COLLECTION" \
EMBED_MODEL="$EMBED_MODEL" \
EMBED_PROVIDER="$EMBED_PROVIDER" \
VECTOR_SIZE="$VECTOR_SIZE" \
SPARSE_MODEL="$SPARSE_MODEL" \
ENABLE_HYBRID="1" \
SEARCH_TIMEOUT="180" \
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
  --query-sleep 21 \
  --out "$RUN_OUT"

echo ""
echo "============================================================"
echo "  Done."
echo "  Summary : $RUN_OUT/summary.json"
echo ""
echo "  Compare against other models:"
LATEST_VOYAGE10=$(find "$REPO_ROOT/tests/eval/runs" -name "summary.json" -path "*voyage_law2_hybrid_10*" | sort | tail -1)
LATEST_BGE=$(find "$REPO_ROOT/tests/eval/runs" -name "summary.json" -path "*bgelarge*" | sort | tail -1)
LATEST_MPNET=$(find "$REPO_ROOT/tests/eval/runs" -name "summary.json" -path "*mpnet_hybrid*" | sort | tail -1)
LATEST_MINILM=$(find "$REPO_ROOT/tests/eval/runs" -name "summary.json" -path "*minilm_hybrid*" | sort | tail -1)
for f in "$LATEST_VOYAGE10" "$LATEST_BGE" "$LATEST_MPNET" "$LATEST_MINILM" "$RUN_OUT/summary.json"; do
  if [ -f "$f" ]; then
    MODEL_LABEL=$(echo "$f" | grep -oP 'cuad_[^/]+(?=/)' | head -1)
    SPAN_HIT1=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"span_hit\"][\"hit@1\"]:.3f}')" 2>/dev/null || echo "?")
    CONTRACT_R20=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"contract_metrics\"][\"recall@20\"]:.3f}')" 2>/dev/null || echo "?")
    echo "  span_hit@1=$SPAN_HIT1  contract_recall@20=$CONTRACT_R20  $MODEL_LABEL"
  fi
done
echo "============================================================"
