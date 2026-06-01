#!/usr/bin/env bash
# run_all_evals.sh
#
# Runs run_eval.py against all 4 active 50-doc model servers and writes
# per-model summary.json files into tests/eval/runs/<collection>_<label>_<date>/.
#
# Prerequisites:
#   bash tests/eval/start_eval_servers.sh   (all 4 servers must be UP first)
#
# Usage:
#   bash tests/eval/run_all_evals.sh
#   bash tests/eval/run_all_evals.sh --label no_reranker
#
# Experiment variables captured in each summary.json:
#   collection, embed_model, embed_provider, vector_size, enable_hybrid,
#   enable_reranker, strategy, top_k, run_timestamp, experiment_label

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="$REPO_ROOT/tests/eval"
RUNS_DIR="$EVAL_DIR/runs"
EVAL_SCRIPT="$EVAL_DIR/run_eval.py"
GOLD_DIR="$EVAL_DIR/gold"

LABEL="${1:-}"
if [ "$LABEL" = "--label" ]; then LABEL="${2:-}"; fi
DATE_SLUG="$(date +%Y%m%d_%H%M)"

# ── Load .env.dev for VOYAGE_API_KEY / HF_TOKEN ────────────────────────────────
ENV_FILE="$REPO_ROOT/.env.dev"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
  echo "Loaded $ENV_FILE"
fi

# ── Model server definitions ──────────────────────────────────────────────────
# Format: "name|port|collection|embed_model|embed_provider|vector_size|query_sleep"
MODELS=(
  "voyage-law-2|8006|cuad_voyage_law2_hybrid_10|voyage-law-2|voyageai|1024|21"
  "bge-large|8013|cuad_bgelarge_hybrid_50|BAAI/bge-large-en-v1.5||1024|0"
  "mpnet|8012|cuad_mpnet_hybrid_50|sentence-transformers/all-mpnet-base-v2||768|0"
  "minilm|8011|cuad_minilm_hybrid_50|sentence-transformers/all-MiniLM-L6-v2||384|0"
)

echo "============================================================"
echo "  run_all_evals.sh"
echo "  Label     : ${LABEL:-<none>}"
echo "  Date      : $DATE_SLUG"
echo "  Models    : ${#MODELS[@]}"
echo "============================================================"
echo ""

RESULTS=()

for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME PORT COLLECTION EMBED_MODEL EMBED_PROVIDER VECTOR_SIZE QUERY_SLEEP <<< "$entry"
  BASE_URL="http://localhost:$PORT"
  GOLD_COLLECTION_DIR="$GOLD_DIR/$COLLECTION"
  LABEL_SUFFIX="${LABEL:+_${LABEL}}"
  OUT_DIR="$RUNS_DIR/${COLLECTION}${LABEL_SUFFIX}_${DATE_SLUG}"

  echo ">>> $NAME  port=$PORT  collection=$COLLECTION"

  # Check server is up
  if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "  [SKIP] $NAME — server not responding on port $PORT"
    RESULTS+=("$NAME: SKIPPED (server down)")
    continue
  fi

  # Check gold exists
  if [ ! -f "$GOLD_COLLECTION_DIR/gold.json" ]; then
    echo "  [SKIP] $NAME — no gold at $GOLD_COLLECTION_DIR"
    RESULTS+=("$NAME: SKIPPED (no gold)")
    continue
  fi

  QDRANT_COLLECTION="$COLLECTION" \
  EMBED_MODEL="$EMBED_MODEL" \
  EMBED_PROVIDER="$EMBED_PROVIDER" \
  VECTOR_SIZE="$VECTOR_SIZE" \
  ENABLE_HYBRID=1 \
    python3 "$EVAL_SCRIPT" \
      --base-url "$BASE_URL" \
      --gold-dir "$GOLD_COLLECTION_DIR" \
      --strategy hybrid_search \
      --query-sleep "$QUERY_SLEEP" \
      --out "$OUT_DIR" \
      ${LABEL:+--experiment-label "$LABEL"}

  echo "  Done → $OUT_DIR/summary.json"
  echo ""
  RESULTS+=("$NAME: $OUT_DIR/summary.json")
done

# ── Print comparison table ──────────────────────────────────────────────────
echo "============================================================"
echo "  Results"
echo "============================================================"
printf "  %-16s  %-8s  %-8s  %-8s  %-8s  %s\n" \
  "model" "R@20" "MRR@10" "MRR@20" "Hit@1" "path"
for entry in "${RESULTS[@]}"; do
  NAME="${entry%%:*}"
  PATH_OR_STATUS="${entry#*: }"
  if [[ "$PATH_OR_STATUS" == *.json ]]; then
    F="$PATH_OR_STATUS"
    R20=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"contract_metrics\"][\"recall@20\"]:.3f}')" 2>/dev/null || echo "?")
    MRR10=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"contract_metrics\"].get(\"mrr@10\",d[\"contract_metrics\"].get(\"mrr@20\",0)):.3f}')" 2>/dev/null || echo "?")
    MRR20=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"contract_metrics\"][\"mrr@20\"]:.3f}')" 2>/dev/null || echo "?")
    HIT1=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"span_hit\"][\"hit@1\"]:.3f}')" 2>/dev/null || echo "?")
    printf "  %-16s  %-8s  %-8s  %-8s  %-8s  %s\n" "$NAME" "$R20" "$MRR10" "$MRR20" "$HIT1" "$F"
  else
    printf "  %-16s  %-8s\n" "$NAME" "$PATH_OR_STATUS"
  fi
done
echo "============================================================"
