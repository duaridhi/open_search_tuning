# Dev-loop latency probes for /search and /chat.
# See readme_docs/SEARCH_LATENCY_PLAN.md for the rollout these baselines gate.
#
# Quick start:
#   make perf STEP=0                    # capture baseline before any change
#   <edit code, restart uvicorn>
#   make perf STEP=1                    # snapshot after step 1
#
# Requires: k6 (https://k6.io/docs/get-started/installation/)
# Requires: uvicorn running locally with PERF_TRACE=1 set in the server env.

STEP        ?= 0
BASE_URL    ?= http://localhost:8000
TOP_K       ?= 10
WARMUP      ?= 2
RUNS        ?= 5
BASELINE_DIR := readme_docs/perf_baselines

.PHONY: perf perf-search perf-chat perf-dir perf-check

perf-dir:
	@mkdir -p $(BASELINE_DIR)

perf-check:
	@command -v k6 >/dev/null 2>&1 || { \
		echo "k6 not found. Install: https://k6.io/docs/get-started/installation/"; exit 1; }
	@curl -sf $(BASE_URL)/health >/dev/null || { \
		echo "Server not reachable at $(BASE_URL). Start uvicorn with PERF_TRACE=1."; exit 1; }

perf-search: perf-check perf-dir
	k6 run \
		-e BASE_URL=$(BASE_URL) -e TOP_K=$(TOP_K) -e WARMUP=$(WARMUP) -e RUNS=$(RUNS) \
		-e OUTFILE=$(BASELINE_DIR)/baseline_step$(STEP)_search.json \
		tests/perf/search.js

perf-chat: perf-check perf-dir
	k6 run \
		-e BASE_URL=$(BASE_URL) -e TOP_K=$(TOP_K) -e WARMUP=$(WARMUP) -e RUNS=$(RUNS) \
		-e OUTFILE=$(BASELINE_DIR)/baseline_step$(STEP)_chat.json \
		tests/perf/chat.js

perf: perf-search perf-chat
	@echo ""
	@echo "Baselines written:"
	@echo "  $(BASELINE_DIR)/baseline_step$(STEP)_search.json"
	@echo "  $(BASELINE_DIR)/baseline_step$(STEP)_chat.json"
