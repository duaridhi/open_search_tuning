---
name: rag-eval
description: Run the CUAD retrieval eval cycle — optionally rebuild gold, run run_eval.py against a live server, and report chunk/contract/span metrics. Use when the user wants to check search quality, catch a regression after a model swap or re-ingest, or compare metrics across runs.
---

# RAG Eval

Runs `build_gold.py` (optional) then `run_eval.py` against a live server, reads
the output `summary.json`, and prints a formatted metrics table with a delta vs.
the previous run.

## When to invoke

- "Run the eval", "Check search quality", "Did that change regress recall?"
- "Run a smoke eval", "Full eval with chat"
- Any explicit `/rag-eval` invocation.

Do NOT invoke for: latency profiling (`/perf`), pre-ingest dataset profiling
(`/eval-dataset`), or Qdrant collection inspection (`qdrant-payload` agent).

## Hard preconditions

Check both before running anything; stop with a clear message if either fails:

1. **Gold files exist.** Check that `tests/eval/gold.json` and
   `tests/eval/gold_contracts.json` are present. If either is missing, tell the
   user and offer to run `build_gold.py` first.
2. **Server reachable.** `curl -sf http://localhost:8000/health` must succeed. If
   not, tell the user to start the server:
   ```
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   and stop.

## Step 1 — (optional) rebuild gold

Only rebuild gold when the user asks, when Qdrant was re-ingested, or when
`gold.json` is absent:

```bash
python tests/eval/build_gold.py
```

Add `--skip-chunks` to only refresh `gold_contracts.json` / `gold_spans.json`
without re-scrolling for chunk-level point IDs (faster, fine after a
non-ingestion change).

## Step 2 — run eval

**Smoke (fast, ~5 queries):**
```bash
python tests/eval/run_eval.py --queries tests/eval/queries_smoke.json
```

**Full (82 queries — use by default unless the user says "smoke"):**
```bash
python tests/eval/run_eval.py
```

**With chat scoring:**
```bash
python tests/eval/run_eval.py --chat
```

**Against a non-default server:**
```bash
python tests/eval/run_eval.py --base-url http://localhost:8001
```

Output goes to `tests/eval/runs/<timestamp>/summary.json` (and `search.json`
for per-query detail).

## Step 3 — read and present results

Read the new `summary.json`. Present a table:

```
Metric                     Current    Prev       Δ
─────────────────────────────────────────────────
contract recall@5          0.412      0.398      +1.4%
contract recall@10         0.531      0.519      +1.2%
contract recall@20         0.614      0.610      +0.4%
contract precision@5       0.083      0.080      +3.7%
contract ndcg@10           0.473      0.461      +2.6%
contract mrr@20            0.321      0.315      +1.9%
─────────────────────────────────────────────────
chunk recall@10            0.038      0.036      +5.6%
span hit@5                 0.744      0.740      +0.5%
span hit@10                0.793      0.789      +0.5%
─────────────────────────────────────────────────
latency p50 (median)       1.42s      1.38s      +2.9%
latency p95 (max)          3.81s      3.65s      +4.4%
```

Pull values from `summary.json`:
- `contract_metrics.{recall@5,recall@10,recall@20,precision@5,ndcg@10,mrr@20}`
- `chunk_metrics.recall@10`
- `span_hit.{hit@5,hit@10}`
- `latency_s.{p50_median_across_queries,p95_max_across_queries}`

**Finding the previous run:** list `tests/eval/runs/` sorted by name (they are
timestamped `YYYY-MM-DD-HHMM`); the second-to-last directory is "prev". If
there's only one run, skip the delta column.

**Regressions:** flag any metric that drops more than 2% relative. Also flag if
`zero_result_queries` is non-empty.

## Step 4 — gold inspection (optional)

After an eval, offer to run the gold dashboard:
```bash
python tests/eval/inspect_gold.py
```

And alignment check after a re-ingest:
```bash
python tests/eval/check_alignment.py
```

## What this skill does NOT do

- It does not start uvicorn. Tell the user if the server isn't running.
- It does not re-run ingestion. Hand off to the `cuad-ingest` agent for that.
- It does not profile latency step-by-step. Use `/perf` for that.
- It does not commit run artifacts. The user decides what to keep.

## File map

| Path | Role |
|---|---|
| `tests/eval/run_eval.py` | Main eval runner |
| `tests/eval/build_gold.py` | Builds gold.json, gold_contracts.json, gold_spans.json |
| `tests/eval/inspect_gold.py` | Stats dashboard for gold artifacts |
| `tests/eval/check_alignment.py` | Qdrant ↔ CUAD gold title alignment check |
| `tests/eval/queries.json` | Full 82-query set |
| `tests/eval/queries_smoke.json` | Smoke subset |
| `tests/eval/gold.json` | Chunk-level ground truth |
| `tests/eval/gold_contracts.json` | Contract-level ground truth |
| `tests/eval/gold_spans.json` | Span-level ground truth |
| `tests/eval/runs/<timestamp>/summary.json` | Eval summary (metrics + latency) |
| `tests/eval/runs/<timestamp>/search.json` | Per-query results |
