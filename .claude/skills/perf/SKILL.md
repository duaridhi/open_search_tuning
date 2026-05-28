---
name: perf
description: Capture a per-step latency baseline for /search and /chat using the k6 dev-loop probes in tests/perf/. Use when the user wants to measure search latency, capture a baseline, check a perf regression after an edit, or compare against a prior step. Knows the PERF_TRACE=1 prerequisite, the canonical CUAD query set, and the snapshot path under readme_docs/perf_baselines/.
---

# Latency dev-loop probe

Runs the k6 probes against a locally running uvicorn, writes a per-step JSON snapshot, and prints a delta vs. the prior step when one exists.

## When to invoke

- "Capture a baseline / capture step N"
- "Run the perf probes"
- "Did that regress search latency?"
- "Diff against step N-1"
- Any explicit `/perf` invocation.

Do NOT invoke for: production observability (that's Grafana Cloud), answer-quality regressions (that's the `rag-eval` agent), or model selection (that's `hf-model-scout`).

## Hard preconditions

Check both before running anything; fail fast with a clear message if either is missing:

1. **k6 installed.** `command -v k6 >/dev/null` — if missing, tell the user to install per https://k6.io/docs/get-started/installation/ and stop.
2. **Server reachable with `PERF_TRACE=1`.** `curl -sf http://localhost:8000/health` must succeed AND the response from `curl -sD - -o /dev/null http://localhost:8000/search?q=ping&top_k=1` must include an `X-Perf-Spans` header. If the header is absent, the server is running without `PERF_TRACE=1`; tell the user to restart with `PERF_TRACE=1 uvicorn app:app --host 0.0.0.0 --port 8000` and stop.

Without `PERF_TRACE=1`, the probes still complete but only `http_req_duration` is populated — span data will be empty and step-over-step comparison loses the breakdown. Don't silently accept that state.

## Step number

If the user named a step, use it. Otherwise pick the next free integer N such that neither `readme_docs/perf_baselines/baseline_step{N}_search.json` nor `..._chat.json` exists. Look in that directory with `ls readme_docs/perf_baselines/ 2>/dev/null`.

## Run

Single command — the Makefile wraps both probes:

```
make perf STEP=$N
```

To run only one side: `make perf-search STEP=$N` or `make perf-chat STEP=$N`.

Optional overrides (only mention these if the user asks):
- `BASE_URL=http://...` (default `http://localhost:8000`)
- `TOP_K=10`
- `WARMUP=2` `RUNS=5`

## After it finishes

1. Read both new snapshot files at `readme_docs/perf_baselines/baseline_step{N}_{search,chat}.json`.
2. If `step{N-1}` snapshots exist for the same endpoint, compute and print a delta table:
   - For each span (`total`, `embed`, `qdrant_query`, `rerank`, `highlight_assemble` for search; `total`, `retrieve`, `chat_completion` for chat), pull `metrics["span_<name>"]["values"]["p(95)"]` from both files and show `prev → curr` plus the percent delta.
   - Flag any span with `>20%` p95 regression.
3. If no prior step exists, just confirm the baseline was captured and list the file paths.

## What this skill does NOT do

- It does not start uvicorn. If the server isn't running, tell the user to start it themselves — auto-starting risks colliding with their existing process and obscures whether `PERF_TRACE=1` is set.
- It does not run answer-quality checks. After a step that touches ranking or generation, hand off to the `rag-eval` agent (see `readme_docs/SEARCH_LATENCY_PLAN.md`).
- It does not commit. The baselines under `readme_docs/perf_baselines/` are checked in once the user is happy with a step.

## File map

| Path | Role |
| --- | --- |
| `Makefile` | `perf`, `perf-search`, `perf-chat`, `perf-check` targets |
| `tests/perf/search.js` | k6 probe for `GET /search` |
| `tests/perf/chat.js` | k6 probe for `POST /chat` |
| `cuad-demo-quadrant/perf_trace.py` | server-side span context manager |
| `readme_docs/perf_baselines/baseline_step{N}_{search,chat}.json` | snapshots |
| `readme_docs/SEARCH_LATENCY_PLAN.md` | the rollout these baselines gate |
