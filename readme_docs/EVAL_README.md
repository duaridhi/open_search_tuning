# CUAD Retrieval Eval

End-to-end retrieval evaluation for the `cuad-ai-demo` search service. Measures
Recall, Precision, MRR, nDCG, and span-hit quality against CUAD ground-truth
annotations, with optional `/chat` source-title coverage.

---

## Directory layout

```
tests/eval/
├── README.md                    ← you are here
├── STRATEGY.md                  ← full design rationale and metric definitions
├── QUERIES.md                   ← query-set schema
├── queries.json                 ← 82 canonical CUAD queries (append-only)
├── queries_smoke.json           ← 8-query smoke set for quick sanity checks
│
├── build_gold.py                ← builds ground-truth files from Qdrant + CUAD annotations
├── run_eval.py                  ← runs queries against /search + /chat, scores metrics
│
├── run_experiment_minilm_50.sh  ← one-shot: ingest → gold → eval (MiniLM, 50 docs)
├── run_experiment_mpnet_100.sh  ← one-shot: ingest → gold → eval (MPNet-768d, 100 docs)
│
├── gold/                        ← per-experiment gold files (gitignored)
│   └── <collection>/
│       ├── gold.json            ← chunk-level: {qid: [point_id, ...]}
│       ├── gold_contracts.json  ← contract-level: {qid: {relevant_titles, ...}}
│       ├── gold_spans.json      ← raw CUAD spans per (qid, title)
│       └── doc_scoped_titles.json
│
└── runs/                        ← eval run output (gitignored)
    └── <collection>_<timestamp>/
        ├── search.json          ← per-query results + metrics
        └── summary.json         ← macro-averaged metrics headline
```

The files `gold.json`, `gold_contracts.json`, `gold_spans.json`, and
`doc_scoped_titles.json` at the top level are the **full-corpus baseline** gold
(all 510 contracts). Experiment-specific gold files live under `gold/<collection>/`.

---

## Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Qdrant must be reachable.
#    Local:  docker run -p 6333:6333 qdrant/qdrant:latest
#    Cloud:  set CLUSTER_URL + QDRANT_API_KEY in cuad-demo-quadrant/.env

# 3. HF_TOKEN must be set (used by app.py for embeddings and chat).
#    Set in cuad-demo-quadrant/.env or as an env var.

# 4. PDF_ROOT must point to the CUAD full_contract_pdf/ directory.
#    Default: /home/ridhi/projects/project1/.../CUAD_v1/full_contract_pdf
#    Override: export PDF_ROOT=/your/path/to/full_contract_pdf
```

---

## Option A — Model comparison experiments (recommended starting point)

Use the one-shot scripts to ingest a sample, build scoped gold, and run eval in
one command. Each script is fully isolated: separate Qdrant collection, separate
gold directory, separate run output. Scripts can be run sequentially or on
separate machines.

### Script 1 — MiniLM-L6-v2, 384-d, 50 contracts

```bash
bash tests/eval/run_experiment_minilm_50.sh
```

What it does:
1. Creates Qdrant collection `cuad_sample_minilm_50` with 384-d cosine vectors (drops and recreates if it exists)
2. Ingests contracts 0–49 (alphabetical by filename) into that collection
3. Builds gold scoped to those 50 contracts → `tests/eval/gold/cuad_sample_minilm_50/`
4. Starts the API server on port **8001** with `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
5. Runs eval → `tests/eval/runs/cuad_sample_minilm_50_<timestamp>/summary.json`

### Script 2 — MPNet-base-v2, 768-d, 100 contracts

```bash
bash tests/eval/run_experiment_mpnet_100.sh
```

Same flow but with `all-mpnet-base-v2` (768-d vectors), 100 contracts, port **8002**.
Results in `tests/eval/runs/cuad_sample_mpnet_100_<timestamp>/summary.json`.

### Customizing a script

All parameters are at the top of each script:

```bash
COLLECTION="cuad_sample_minilm_50"   # Qdrant collection name
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE=384                       # must match the model's output dimension
DOC_OFFSET=0                          # skip first N PDFs
DOC_COUNT=50                          # how many PDFs to ingest (0 = all)
SERVER_PORT=8001                      # port for the temporary API server
```

`VECTOR_SIZE` is used to create the Qdrant collection (step 1). It must match the
embedding model's actual output dimension — if they disagree, Qdrant will reject
vectors during ingest. Common values: `384` (MiniLM-L6-v2, bge-small-en-v1.5),
`768` (mpnet-base-v2, bge-base-en-v1.5), `1024` (bge-large-en-v1.5).

Change any of these, save, and re-run.

---

## Option B — Manual step-by-step

Use this when you want to re-run just one stage (e.g. re-eval after a code change
without re-ingesting).

### Step 1 — Ingest a sample collection

```bash
QDRANT_COLLECTION=cuad_sample_minilm_50 \
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
DOC_OFFSET=0 \
DOC_COUNT=50 \
SKIP_INGESTED_DOCS=0 \
  python cuad-demo-quadrant/upload_to_qdrant.py
```

Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_COLLECTION` | `cuad_contracts` | Collection to create/upsert into |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name or HF path |
| `DOC_OFFSET` | `0` | Skip first N PDFs (alphabetical order) |
| `DOC_COUNT` | `0` (all) | Number of PDF files to process |
| `SKIP_INGESTED_DOCS` | `1` | Set to `0` to force re-ingest of all docs |
| `PDF_ROOT` | (hardcoded path) | Path to `full_contract_pdf/` |

`VECTOR_SIZE` is derived automatically from the model — no need to set it manually.

### Step 2 — Build gold

```bash
python tests/eval/build_gold.py \
  --collection cuad_sample_minilm_50 \
  --out-dir    tests/eval/gold/cuad_sample_minilm_50
```

This scrolls all points in the collection, joins them against CUAD annotations
downloaded from HuggingFace (`theatticusproject/cuad`), and writes:

- `gold.json` — chunk-level relevant point IDs per query
- `gold_contracts.json` — relevant contract titles per query  
- `gold_spans.json` — raw annotated spans per (query, title)
- `doc_scoped_titles.json` — 5-title random sample for document-scoped queries

Omit `--out-dir` to write to `tests/eval/` (overwrites the full-corpus baseline).

Full rebuild vs. contracts-only refresh:

```bash
# Rebuild everything (slow — re-scrolls Qdrant and re-downloads CUAD)
python tests/eval/build_gold.py --collection <name> --out-dir <dir>

# Refresh only gold_contracts.json + gold_spans.json (skips chunk-level scroll)
python tests/eval/build_gold.py --collection <name> --out-dir <dir> --skip-chunks
```

### Step 3 — Start the API server

```bash
QDRANT_COLLECTION=cuad_sample_minilm_50 \
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  uvicorn app:app --host 0.0.0.0 --port 8001
```

**`EMBED_MODEL` must match what was used for ingest.** The server uses it to embed
queries at search time — a mismatch silently produces wrong similarity scores.

### Step 4 — Run eval

```bash
python tests/eval/run_eval.py \
  --base-url http://localhost:8001 \
  --gold-dir tests/eval/gold/cuad_sample_minilm_50 \
  --out      tests/eval/runs/my_run
```

All flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8000` | API server to test |
| `--gold-dir` | `tests/eval/` | Directory with gold files from `build_gold.py` |
| `--out` | `tests/eval/runs/<timestamp>/` | Output directory |
| `--queries` | `tests/eval/queries.json` | Query file to use |
| `--chat` | off | Also hit `/chat` and check source-title coverage |

For a quick smoke test (8 queries, ~1 min):

```bash
python tests/eval/run_eval.py \
  --queries  tests/eval/queries_smoke.json \
  --base-url http://localhost:8001 \
  --gold-dir tests/eval/gold/cuad_sample_minilm_50
```

---

## Reading results

After `run_eval.py` completes, open `<out>/summary.json`:

```jsonc
{
  "n_queries": 82,
  "chunk_metrics": {
    "recall@10": 0.42,   // fraction of gold chunks found in top-10
    "mrr@20": 0.61,      // mean reciprocal rank of first relevant chunk
    "ndcg@10": 0.55
  },
  "contract_metrics": {
    "recall@10": 0.58,   // fraction of gold contracts surfaced in top-10
    "mrr@20": 0.72
  },
  "span_hit": {
    "hit@1":  0.39,      // gold answer text found in top-1 result
    "hit@5":  0.61,
    "hit@10": 0.68,
    "mrr":    0.51
  },
  "latency_s": {
    "p50_median_across_queries": 1.2,
    "p95_max_across_queries": 4.8
  },
  "zero_result_queries": []  // any query returning 0 results — investigate immediately
}
```

Per-query detail is in `<out>/search.json` — useful for diagnosing which categories
are weak. Open it and filter by `"recall@10": 0` to find the worst queries.

### Comparing two runs

```bash
# Quick diff of headline metrics
python - <<'EOF'
import json, sys
a = json.load(open("tests/eval/runs/cuad_sample_minilm_50_20260529_1400/summary.json"))
b = json.load(open("tests/eval/runs/cuad_sample_mpnet_100_20260529_1500/summary.json"))
for k in ["recall@10", "mrr@20", "ndcg@10"]:
    va = a["chunk_metrics"][k]; vb = b["chunk_metrics"][k]
    print(f"{k:20s}  minilm={va:.3f}  mpnet={vb:.3f}  delta={vb-va:+.3f}")
EOF
```

---

## Full-corpus baseline (all 510 contracts)

The files `tests/eval/gold.json`, `gold_contracts.json`, and `gold_spans.json` at
the top level are the full-corpus baseline gold. To rebuild them after a re-ingest:

```bash
# Assumes the full corpus is already in cuad_contracts (the default collection)
python tests/eval/build_gold.py
```

No `--collection` or `--out-dir` needed — both default to `cuad_contracts` and
`tests/eval/` respectively.

---

## Known issues

- **40 contracts have truncation gaps** — trailing clause spans fall outside the
  ingested character range. See [INGEST_HANDOFF.md](INGEST_HANDOFF.md) for the
  full list and remediation steps. This suppresses Recall on affected categories
  (primarily *Governing Law*).
- **Gold is scoped to ingested titles** — if you run `build_gold.py` against a
  sample collection, Recall denominators are correct *within that sample* but
  cannot be directly compared against full-corpus numbers.
- **CUAD annotations download on first run** — `build_gold.py` fetches
  `CUAD_v1.json` from HuggingFace (~30 MB) and caches it in
  `tests/eval/cuad_gold/_hf_cache/`. Subsequent runs use the cache.
