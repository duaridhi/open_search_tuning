# CUAD Retrieval Experiments — Results & Analysis

Comprehensive record of all retrieval experiments conducted on the CUAD legal contract corpus. Intended for anyone picking up this project without prior context.

---

## 1. Background

### Dataset

**CUAD** (Contract Understanding Atticus Dataset) — 510 legal contracts covering 41 clause categories. Each contract is annotated with expert-labeled answer spans for every applicable category.

| Stat | Value |
|---|---|
| Total contracts | 510 |
| Clause categories | 41 |
| Annotated Q&A pairs | ~13,000 |
| PDF source | HuggingFace dataset repo |

### What we're building

A semantic search API over the corpus:
- PDFs are chunked into overlapping text segments and embedded with a dense vector model
- Optionally augmented with sparse BM42 vectors for hybrid (dense + sparse) retrieval via Qdrant RRF fusion
- At query time, the top-k most similar chunks are returned and reranked by sentence for highlighting

### Repo layout

```
app.py                             FastAPI server (active)
cuad-demo-quadrant/
  upload_to_qdrant_hf.py           Ingest: PDF → chunks → embeddings → Qdrant
  qdrant_search_hf.py              Query: embed → Qdrant search → sentence rerank
  chat_hf.py                       RAG: top-k chunks → Qwen3 grounded answer
tests/eval/
  build_gold.py                    Build gold.json / gold_contracts.json per collection
  run_eval.py                      Eval: fire 82 queries, compute metrics, write summary.json
  compare_models.py                Per-document multi-model comparison script (interactive)
  run_experiment_*.sh              One-shot scripts: ingest + gold + server + eval per model
  runs/                            Eval output — one timestamped dir per run
```

---

## 2. Evaluation Methodology

### Query set

`tests/eval/queries.json` — 82 queries, one per active (category, variant) combination. Two variant types:
- `__question`: Full natural-language CUAD question with clause definition (used for experiments below unless noted)
- `__short`: Short keyword-style query (earlier experiments)

### Gold artifacts (per collection)

`build_gold.py` builds three files scoped to the specific Qdrant collection:

| File | Granularity | How built |
|---|---|---|
| `gold.json` | Chunk (Qdrant point IDs) | Text substring match: CUAD answer span found in chunk text |
| `gold_contracts.json` | Contract (title) | Which contract titles contain a relevant annotation |
| `gold_spans.json` | Raw text spans | CUAD answer text per (query, document) pair |

### Metrics

**Chunk-level** (how well do retrieved chunks match gold answer chunks):
- `recall@k` — fraction of gold chunks appearing in top-k results (macro-averaged across queries)
- `precision@k` — fraction of top-k results that are gold chunks
- `mrr@20` — mean reciprocal rank of first gold chunk
- `ndcg@10` — normalized discounted cumulative gain at 10

**Contract-level** (at what rate is the correct contract retrieved):
- Same metrics but over contract titles, not individual chunks
- More lenient: any chunk from the right contract counts

**Span-level hit rate** (did a gold annotation appear in retrieved text):
- `hit@k` — fraction of queries where any gold span text appears in top-k chunks
- `mrr` — mean reciprocal rank of first span hit

**Latency**: p50 and p95 of HTTP round-trip from 5 repeat runs per query.

### Infrastructure

- **Qdrant Cloud**: `eu-west-1`, cosine similarity, hybrid (dense + sparse RRF)
- **Ingest**: `upload_to_qdrant_hf.py` — `pdfplumber` extraction, overlapping character chunks, VoyageAI or HF Inference API embedding, BM42 sparse via `fastembed`
- **Server**: FastAPI on port 8000+ with `asyncio.to_thread` wrapping, sentence-level reranking for highlights
- **Eval client**: `run_eval.py` fires queries sequentially with `--query-sleep` to respect rate limits

---

## 3. Experiments Conducted

### Naming convention

Collections follow `cuad_{model}_{hybrid_}{N}` where `N` is the overlap (for 500-char chunks) or chunk size indicator.

---

### Experiment E0a — MiniLM-L6-v2, 500-char chunks, dense-only (baseline)

| Parameter | Value |
|---|---|
| Collection | `cuad_sample_minilm_50` |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Sparse model | None (dense-only) |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_minilm_50.sh` |
| Run date | 2026-05-29 |
| Run dir | `runs/cuad_sample_minilm_50_fixed_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.073 / 0.100 / 0.144 |
| chunk MRR@20 | 0.244 |
| chunk nDCG@10 | 0.132 |
| contract recall@5 / @10 / @20 | 0.160 / 0.254 / 0.391 |
| contract MRR@20 | 0.678 |
| span hit@1 / @5 / @10 | 0.115 / 0.500 / 0.590 |
| span MRR | 0.407 |
| latency p50 / p95 | 0.168s / 1.79s |

*Dense-only baseline. Direct comparison with E1 (same model + hybrid) shows the additive value of BM42 sparse.*

---

### Experiment E0b — MPNet-base-v2, 500-char chunks, dense-only (baseline)

| Parameter | Value |
|---|---|
| Collection | `cuad_sample_mpnet_50` |
| Dense model | `sentence-transformers/all-mpnet-base-v2` |
| Dimensions | 768 |
| Sparse model | None (dense-only) |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_mpnet_50.sh` |
| Run date | 2026-05-29 |
| Run dir | `runs/cuad_sample_mpnet_50_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.089 / 0.119 / 0.150 |
| chunk MRR@20 | 0.251 |
| chunk nDCG@10 | 0.150 |
| contract recall@5 / @10 / @20 | 0.181 / 0.270 / 0.398 |
| contract MRR@20 | 0.792 |
| span hit@1 / @5 / @10 | 0.128 / 0.462 / 0.615 |
| span MRR | 0.382 |
| latency p50 / p95 | 0.173s / 2.43s |

*Dense-only baseline. Direct comparison with E2 (same model + hybrid) shows the additive value of BM42 sparse.*

---

### Experiment E1 — MiniLM-L6-v2, 500-char chunks, hybrid

| Parameter | Value |
|---|---|
| Collection | `cuad_minilm_hybrid_50` |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Sparse model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_minilm_hybrid_50.sh` |
| Run date | 2026-05-30 |
| Run dir | `runs/cuad_minilm_hybrid_50_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.096 / 0.126 / 0.174 |
| chunk MRR@20 | 0.246 |
| chunk nDCG@10 | 0.147 |
| contract recall@5 / @10 / @20 | 0.168 / 0.266 / 0.405 |
| contract MRR@20 | 0.698 |
| span hit@1 / @5 / @10 | 0.103 / 0.474 / 0.603 |
| span MRR | 0.372 |
| latency p50 / p95 | 0.190s / 2.32s |

---

### Experiment E2 — MPNet-base-v2, 500-char chunks, hybrid

| Parameter | Value |
|---|---|
| Collection | `cuad_mpnet_hybrid_50` |
| Dense model | `sentence-transformers/all-mpnet-base-v2` |
| Dimensions | 768 |
| Sparse model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_mpnet_hybrid_50.sh` |
| Run date | 2026-05-30 |
| Run dir | `runs/cuad_mpnet_hybrid_50_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.094 / 0.124 / 0.183 |
| chunk MRR@20 | 0.248 |
| chunk nDCG@10 | 0.149 |
| contract recall@5 / @10 / @20 | 0.179 / 0.269 / 0.418 |
| contract MRR@20 | 0.750 |
| span hit@1 / @5 / @10 | 0.128 / 0.449 / 0.603 |
| span MRR | 0.361 |
| latency p50 / p95 | 0.190s / 2.50s |

---

### Experiment E3 — BGE-Large-EN-v1.5, 500-char chunks, hybrid

| Parameter | Value |
|---|---|
| Collection | `cuad_bgelarge_hybrid_50` |
| Dense model | `BAAI/bge-large-en-v1.5` |
| Dimensions | 1024 |
| Sparse model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_bgelarge_hybrid_50.sh` |
| Run date | 2026-05-30 |
| Run dir | `runs/cuad_bgelarge_hybrid_50_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.094 / 0.124 / 0.187 |
| chunk MRR@20 | 0.299 |
| chunk nDCG@10 | 0.162 |
| contract recall@5 / @10 / @20 | 0.173 / 0.278 / 0.391 |
| contract MRR@20 | 0.748 |
| span hit@1 / @5 / @10 | 0.179 / 0.474 / 0.551 |
| span MRR | 0.441 |
| latency p50 / p95 | 0.192s / 3.41s |

---

### Experiment E4 — Qwen2 (first 10 docs), 500-char chunks

| Parameter | Value |
|---|---|
| Collection | `cuad_qwen2_10` |
| Dense model | Qwen2-based embedding |
| Chunk size | 500 chars, 50 overlap |
| Documents | 10 (offset 0) |
| Script | `run_experiment_qwen2_10.sh` |
| Run date | 2026-05-30 |

**Results (10-doc scope, not directly comparable):**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.081 / 0.123 / 0.165 |
| chunk MRR@20 | 0.315 |
| contract recall@5 / @10 / @20 | — / 0.253 / 0.385 |
| contract MRR@20 | 0.789 |
| span hit@1 / @5 / @10 | 0.218 / 0.449 / 0.590 |
| span MRR | 0.478 |

*Note: Only 10 docs ingested. Higher span MRR and contract MRR suggests Qwen2 has strong ranking within the narrow scope but recall is limited by the 10-doc corpus.*

---

### Experiment E5 — voyage-law-2, 500-char chunks, hybrid (primary baseline)

| Parameter | Value |
|---|---|
| Collection | `cuad_voyage_law2_hybrid_10` |
| Dense model | `voyage-law-2` (VoyageAI, legal domain fine-tuned) |
| Dimensions | 1024 |
| Sparse model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| Chunk size | 500 chars, 50 overlap |
| Documents | 50 (offset 0) |
| Script | `run_experiment_voyage_law2_hybrid_10.sh` |
| Run date | 2026-05-31 |
| Run dir | `runs/cuad_voyage_law2_hybrid_10_voyage-law-2_.../` |

**Results:**

| Metric | Value |
|---|---|
| chunk recall@5 / @10 / @20 | 0.100 / 0.159 / 0.231 |
| chunk MRR@20 | 0.365 |
| chunk nDCG@10 | 0.199 |
| contract recall@5 / @10 / @20 | 0.204 / 0.315 / 0.435 |
| contract MRR@20 | 0.762 |
| span hit@1 / @5 / @10 | 0.244 / 0.526 / 0.641 |
| span MRR | 0.493 |
| latency p50 / p95 | 0.203s / 2.79s |

*Collection name `_10` is a legacy naming artifact; it actually covers 50 documents.*

---

### Experiment E6 — voyage-law-2, 200-char chunks, hybrid (PENDING)

| Parameter | Value |
|---|---|
| Collection | `cuad_voyage_law2_hybrid_200` |
| Dense model | `voyage-law-2` |
| Dimensions | 1024 |
| Sparse model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| Chunk size | **200 chars**, 50 overlap |
| Documents | 50 (offset 0) — **1 ingested so far** (DigitalCinema) |
| Script | `run_experiment_voyage_law2_hybrid_200.sh` |
| Status | **Pending — run script to complete ingest (~3 h) and eval** |

*Purpose: Test whether smaller chunks improve recall by reducing embedding dilution (a single annotation buried in a 500-char chunk about other content gets its own focused vector at 200 chars).*

*Estimated total run time: ~4 h (3 h ingest + 30 min gold + 30 min eval).*

---

### Experiment E7 — Cross-encoder result reranking (voyage-law-2 & bge-large)

| Parameter | Value |
|---|---|
| Collections | `cuad_voyage_law2_hybrid_10` (voyage), `cuad_bgelarge_hybrid_50` (bge) |
| Embedders | `voyage-law-2` (1024) / `BAAI/bge-large-en-v1.5` (1024) |
| Retrieval | hybrid (dense + BM42 sparse, RRF) — same as E5 / E3 |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, ~80 MB) |
| Candidate pool | `top_k × RERANK_POOL` = 20 × 5 = **100 chunks** reranked → top-20 |
| Toggle | `RERANK_RESULTS=1` in [qdrant_search_hf.py](../../cuad-demo-quadrant/qdrant_search_hf.py) |
| Run date | 2026-05-31 |
| Run dirs | `runs/cuad_voyage_law2_hybrid_10_rerank_.../`, `runs/cuad_bgelarge_hybrid_50_rerank_.../` |

This is a **true result-list reranker** (reorders which chunks rank first), distinct from
the `ENABLE_RERANKER` highlight path (which only scores sentences *within* an already-ranked
chunk and never changes result order). Hardware (5 GB RAM, no GPU) ruled out
`BAAI/bge-reranker-v2-m3`; the small MS-MARCO MiniLM cross-encoder was used instead.

**Results vs baseline (E5 voyage hybrid / E3 bge hybrid, both no reranker):**

```
voyage-law-2            baseline(E5)  +rerank(E7)    Δ          bge-large          baseline(E3)  +rerank(E7)    Δ
──────────────────────────────────────────────────────       ──────────────────────────────────────────────────────
chunk recall@10           0.159        0.173       +9%         chunk recall@10       0.124        0.178       +44%
chunk recall@20           0.231        0.235       +2%         chunk recall@20       0.187        0.237       +27%
chunk MRR@20              0.365        0.376       +3%         chunk MRR@20          0.299        0.395       +32%
chunk nDCG@10             0.199        0.241      +21%         chunk nDCG@10         0.162        0.250       +54%
contract MRR@20           0.762        0.831       +9%         contract MRR@20       0.748        0.878       +17%
span hit@1                0.244        0.256       +5%         span hit@1            0.179        0.269       +50%
span hit@10               0.641        0.667       +4%         span hit@10           0.551        0.705       +28%
span MRR                  0.493        0.526       +7%         span MRR              0.441        0.534       +21%
latency p50               0.203s       7.27s       36×         latency p50           0.192s       8.46s       44×
──────────────────────────────────────────────────────       ──────────────────────────────────────────────────────
```

*Caveat:* contract recall@10 dipped for voyage (0.315 → 0.285) — the reranker optimizes for
putting the single best chunk first (↑ MRR, ↑ hit@1) at the slight expense of pool breadth
(fewer distinct relevant contracts survive in top-10). Contract recall@20 held flat.

---

### Experiment E7b — Reranker latency vs candidate-pool size

Follow-up to E7 isolating *where* the rerank latency comes from and how it scales with the
candidate pool (`RERANK_POOL`). Two measurements on the same 4-core CPU box (no GPU):

**1. Isolated reranker forward pass** — `cross-encoder/ms-marco-MiniLM-L-6-v2` `.predict()`
on N `(query, ~500-char chunk)` pairs, batch_size=32, warm, best of 3:

| Pool (pairs) | Rerank compute | Throughput |
|---|---|---|
| 25 | 0.41 s | ~61 pairs/s |
| **50** | **1.17 s** | ~43 pairs/s |
| 100 | 2.03 s | ~49 pairs/s |

Near-linear at ~50 pairs/s. Halving the pool 100 → 50 saves ~0.8 s of model compute.

**2. End-to-end server latency** — the ~7–8 s p50 reported in E7 is **not** the rerank math
(only ~2 s for 100 pairs). It was dominated by **CPU contention**: the E7 eval ran two model
servers (voyage 8006 + bge 8013) competing for 4 cores, plus cold-cache effects. Server-side
`top_k=10` (pool 50) vs `top_k=20` (pool 100) probes came out noisy and sometimes inverted
(5–15 s either way), confirming contention — not pool size — set the wall-clock time.

**Practical latency by scenario** (pool 50 vs 100):

| Scenario | Pool 50 | Pool 100 |
|---|---|---|
| Quiet box / single server (embed ~0.3 s + rerank) | ~1.5 s | ~2.3 s |
| Loaded (two servers, 4 cores) — as in E7 | ~5–6 s | ~7–8 s |

**Takeaways:**
- Pool size scales rerank compute predictably (~0.8 s saved per 50 candidates dropped), but on
  this hardware it is a **secondary** knob — CPU contention is the dominant latency driver.
  Bigger levers: one server at a time, a GPU, or a distilled/ONNX reranker.
- To deploy with a 50-chunk pool, set `RERANK_POOL` so `top_k × RERANK_POOL = 50`
  (e.g. `top_k=10, RERANK_POOL=5`).

---

## 4. Cross-Model Comparison (50-doc runs, comparable scope)

### 4a. Hybrid search — E1–E3 and E5

All four models using 50 docs, 500-char chunks, hybrid (dense + BM42 sparse, RRF fusion).

```
Model                     dim  Chunk  ck-r@10  ck-r@20  ck-MRR  ct-r@10  ct-r@20  ct-MRR  sh@1   sh@10  sMRR   p50-lat
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
voyage-law-2 (hybrid)    1024   500    0.159    0.231    0.365    0.315    0.435    0.762   0.244  0.641  0.493  0.203s
bge-large-en (hybrid)    1024   500    0.124    0.187    0.299    0.278    0.391    0.748   0.179  0.551  0.441  0.192s
mpnet-base-v2 (hybrid)    768   500    0.124    0.183    0.248    0.269    0.418    0.750   0.128  0.603  0.361  0.190s
MiniLM-L6-v2 (hybrid)     384   500    0.126    0.174    0.246    0.266    0.405    0.698   0.103  0.603  0.372  0.190s
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
*ck = chunk, ct = contract, sh = span hit, sMRR = span MRR*

### 4b. Dense-only vs hybrid ablation — E0a/E0b vs E1/E2

Direct comparison of the same model with and without BM42 sparse vectors. All other parameters held constant (50 docs, 500-char chunks).

```
Model                       ck-r@10   Δ    ck-r@20   Δ     ct-r@10   Δ    ct-r@20   Δ     sh@1    Δ
──────────────────────────────────────────────────────────────────────────────────────────────────────
MiniLM dense-only (E0a)      0.100         0.144          0.254          0.391          0.115
MiniLM hybrid     (E1)       0.126  +26%   0.174   +21%   0.266   +5%   0.405   +4%   0.103  −10%
──────────────────────────────────────────────────────────────────────────────────────────────────────
MPNet  dense-only (E0b)      0.119         0.150          0.270          0.398          0.128
MPNet  hybrid     (E2)       0.124   +4%   0.183   +22%   0.269    0%   0.418   +5%   0.128    0%
──────────────────────────────────────────────────────────────────────────────────────────────────────
```
*ck = chunk-level, ct = contract-level, sh@1 = span hit@1*

**Key takeaway**: BM42 hybrid adds ~+21% chunk recall but provides no benefit (and slight regression) on span hit@1. The BM42 lexical signal improves how many relevant chunks enter the top-20 pool, but the ranking within that pool is unchanged — so the first result is no more likely to contain the exact gold span.

---

## 5. Key Findings

### F1 — voyage-law-2 leads on all primary metrics

voyage-law-2 outperforms every general-purpose model on chunk recall, chunk MRR, and span hit rate:
- **chunk recall@20**: +23% over bge-large (0.231 vs 0.187)
- **span hit@1**: +36% over bge-large (0.244 vs 0.179); +136% over MiniLM (0.244 vs 0.103)
- **span hit@10**: +16% over bge-large (0.641 vs 0.551)

The legal-domain fine-tuning of voyage-law-2 gives it a large advantage for CUAD clause retrieval, especially at hit@1 (first-result precision) which matters most for interactive use.

### F2 — Contract-level recall is similar across models

contract recall@20 ranges from 0.391 (bge-large) to 0.435 (voyage-law-2), a tighter 11% gap vs the 23% gap at chunk level. This means all models can identify *which contract* to look at, but voyage-law-2 is better at surfacing the *exact passage*.

### F3 — MiniLM-L6-v2 is competitive on contract-level, not span-level

MiniLM has contract MRR 0.698 vs voyage-law-2's 0.762 (−8%), which is a much smaller gap than its span hit@1 of 0.103 vs 0.244 (−58%). MiniLM is adequate for document-level retrieval (finding the right contract) but poor at exact clause location.

### F4 — Chunk-level recall is universally low

All models have chunk recall@20 below 0.25. This is expected: CUAD has 41 categories × 50 contracts = up to 2050 relevant chunks, but the corpus at 500-char chunks has ~7000–8000 chunks total. Most queries have many relevant chunks spread through the document, and top-20 can only surface a small fraction.

The practical metric for the product is **span hit@10** (did the user see the right text in the first page of results?) which ranges from 0.551 to 0.641.

### F5 — The 0.80 recall ceiling was an eval bug

Earlier analysis of a single document (DigitalCinema) appeared to show Ann@20 = 0.80 as a hard ceiling. This was an artifact of the eval using exact substring match for gold spans — when spans are longer than the chunk size (200-char chunks with 250-char spans), no single chunk contains the full span text. After fixing to use sliding-window partial matching, the correct ceiling is **Ann@50 = 0.96** for all models.

The genuine missing span (the 4%) is *"More detailed quality audits may be performed by NCM personnel."* — a 62-char sentence about quality procedures embedded in a quality-standards section. Its chunk embedding is semantically about quality specs, not audit rights, so no audit-rights query surfaces it via dense retrieval.

### F8 — Hybrid (BM42) improves chunk recall but not span precision

Comparing dense-only (E0a/E0b) against hybrid (E1/E2) with all other parameters fixed:

- **chunk recall@20**: +21–22% with hybrid (more relevant chunks surface in top-20)
- **contract recall@20**: +4–5% with hybrid (marginal improvement)
- **span hit@1**: 0% or −10% with hybrid (first result is no more likely to have the gold span)
- **Latency**: no measurable overhead from hybrid (RRF fusion is server-side in Qdrant)

The BM42 sparse model adds lexical keyword recall — it promotes chunks that contain exact query terms that the dense model might miss. This widens the recall pool but does not improve result precision (which chunk ranks first). If your use case is "find any relevant chunk" (recall-oriented), hybrid helps. If your use case is "the top result should contain the exact answer" (precision-oriented), hybrid alone is insufficient and a cross-encoder reranker over the candidate pool would be needed.

### F9 — Cross-encoder reranking is the biggest single precision lever (E7)

Adding a CPU cross-encoder reranker over the top-100 candidate pool (re-sort → top-20) is
the largest quality gain of any change tested, and it improves both chunk-level and span-level
metrics — exactly the dimensions that matter most for this product:

- **bge-large**: span hit@1 **+50%** (0.179 → 0.269), chunk MRR@20 **+32%**, nDCG@10 **+54%**.
- **voyage-law-2**: span hit@1 +5%, chunk nDCG@10 +21%, contract MRR@20 +9%.

Two structural conclusions:

1. **The reranker mostly erases the embedder gap.** Reranked bge-large (span hit@1 0.269,
   chunk recall@20 0.237, nDCG@10 0.250) now **matches or slightly beats** reranked
   voyage-law-2 (0.256 / 0.235 / 0.241). The same cross-encoder dictates final order, so the
   choice of dense embedder matters far less once a reranker sits on top — it only needs to
   pull the right chunk into the 100-candidate pool. A cheap local reranker on a general
   embedder beats a premium legal embedder used alone.
2. **voyage gains less because its raw ordering was already good** (legal-domain tuning).
   bge gains more because it had more mis-ordering for the reranker to fix.

**Cost**: latency p50 jumps from ~0.2 s to **7–8 s** on CPU (100 pairs through MS-MARCO
MiniLM, no GPU). This is the dominant deployment caveat. Mitigations: smaller pool
(`RERANK_POOL=3`), GPU inference, a distilled/ONNX reranker, or rerank only top-K×2.
This validates the F8 hypothesis: hybrid widens the recall pool, and a reranker is what
converts that pool into first-result precision.

### F6 — Multi-query (query decomposition) did not improve recall

An experiment firing 2–3 sub-queries per multi-annotation category (License Grant, Audit Rights, Anti-Assignment) and union-merging results showed zero improvement over the single-query baseline. The missing annotations were not surfaced by any of the targeted sub-queries. The bottleneck is embedding dilution at the chunk level, not query formulation.

### F7 — Smaller chunks (200-char) do not improve recall at top-50

The single-document DigitalCinema pilot with 200-char chunks showed:
- Ann@50: same 0.96 as 500-char collection
- Ann@20: **worse** (0.92 vs 0.96) because top-50 covers a smaller fraction of 808 chunks than of ~160 chunks

The revenue/profit rank improved (4→1) and anti-assignment ranks tightened, but overall recall at the practical operating k (top-10 to top-20) is similar or slightly worse due to the larger total chunk count.

*Full 50-doc eval for the 200-char collection is pending (Experiment E6).*

---

## 6. Deep-Dive: DigitalCinema Document Analysis

Two compare_models.py runs were saved against `DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement` — one with short-form queries and one with full CUAD question-form queries. Both use hybrid_search, top-20.

Raw output saved in: `tests/eval/compare_models_digital_cinema.txt` (short query) and `tests/eval/compare_models_digital_cinema_questions.txt` (question form).

---

### 6a. Run 1 — Short query form

**Retrieval summary** (rank of first gold span hit; `-` = miss in top-20):

| Model | Anti-Assignment | License Grant | Governing Law | Audit Rights | Revenue/Profit | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|---|---|---|
| voyage-law-2 | 1 ✓ | 6 ✓ | 1 ✓ | 2 ✓ | 3 ✓ | 0.40 | 0.80 | 0.600 |
| bge-large | 1 ✓ | 6 ✓ | 1 ✓ | 1 ✓ | 2 ✓ | 0.60 | 0.80 | 0.733 |
| mpnet | 1 ✓ | 3 ✓ | 2 ✓ | 3 ✓ | 2 ✓ | 0.20 | 1.00 | 0.533 |
| minilm | 1 ✓ | 3 ✓ | 1 ✓ | 1 ✓ | 2 ✓ | 0.60 | 1.00 | 0.767 |

**Annotation recall** (fraction of gold spans found in top-k chunks):

| Model | Anti-Assignment | License Grant | Governing Law | Audit Rights | Revenue/Profit | Ann@5 | Ann@10 | Ann@20 |
|---|---|---|---|---|---|---|---|---|
| voyage-law-2 | 3/3 | 0/5 | 1/1 | 2/5 | 1/1 | 0.68 | 0.80 | 0.80 |
| bge-large | 3/3 | 0/5 | 1/1 | 2/5 | 1/1 | 0.68 | 0.80 | 0.80 |
| mpnet | 2/3 | 1/5 | 1/1 | 2/5 | 1/1 | 0.65 | 0.73 | 0.80 |
| minilm | 2/3 | 1/5 | 1/1 | 2/5 | 1/1 | 0.65 | 0.69 | 0.80 |

---

### 6b. Run 2 — Full CUAD question form

**Retrieval summary**:

| Model | Anti-Assignment | License Grant | Governing Law | Audit Rights | Revenue/Profit | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|---|---|---|
| voyage-law-2 | 1 ✓ | 7 ✓ | 1 ✓ | 1 ✓ | 4 ✓ | 0.60 | 0.80 | 0.679 |
| bge-large | — | 14 ✓ | 1 ✓ | 1 ✓ | 5 ✓ | 0.40 | 0.60 | 0.454 |
| mpnet | 2 ✓ | 4 ✓ | 1 ✓ | 1 ✓ | 6 ✓ | 0.40 | 0.80 | 0.583 |
| minilm | 3 ✓ | 9 ✓ | 1 ✓ | 1 ✓ | 4 ✓ | 0.40 | 0.80 | 0.539 |

**Annotation recall**:

| Model | Anti-Assignment | License Grant | Governing Law | Audit Rights | Revenue/Profit | Ann@5 | Ann@10 | Ann@20 |
|---|---|---|---|---|---|---|---|---|
| voyage-law-2 | 3/3 | 0/5 | 1/1 | 2/5 | 1/1 | 0.68 | 0.80 | 0.80 |
| bge-large | — | 0/5 | 1/1 | 3/5 | 1/1 | 0.65 | 0.65 | 0.75 |
| mpnet | 3/3 | 1/5 | 1/1 | 2/5 | 0/1 | 0.52 | 0.72 | 0.80 |
| minilm | 2/3 | 0/5 | 1/1 | 2/5 | 1/1 | 0.61 | 0.76 | 0.80 |

---

### 6c. Per-query annotation recall detail (question-form run)

| Query | Gold spans | voyage-law-2 | bge-large | mpnet | minilm |
|---|---|---|---|---|---|
| Anti-Assignment | 3 | Rank 1 / 3/3@5 / 3/3@20 | Rank — / — | Rank 2 / 3/3@5 / 3/3@20 | Rank 3 / 2/3@5 / 3/3@20 |
| License Grant | 5 | Rank 7 / 0/5@5 / 2/5@20 | Rank 14 / 0/5@5 / 2/5@20 | Rank 4 / 1/5@5 / 2/5@20 | Rank 9 / 0/5@5 / 2/5@20 |
| Governing Law | 1 | Rank 1 / 1/1@1 | Rank 1 / 1/1@1 | Rank 1 / 1/1@1 | Rank 1 / 1/1@1 |
| Audit Rights | 5 | Rank 1 / 2/5@5 / 3/5@20 | Rank 1 / 3/5@5 / 3/5@20 | Rank 1 / 2/5@5 / 3/5@20 | Rank 1 / 2/5@5 / 3/5@20 |
| Revenue/Profit | 1 | Rank 4 / 1/1@5 | Rank 5 / 1/1@5 | Rank 6 / 0/1@5 / 1/1@10 | Rank 4 / 1/1@5 |

**Key observations**:
- Governing Law: all models hit rank 1 — unambiguous clause with unique vocabulary
- License Grant: only 2/5 annotations recovered at top-20 by all models — the 3 missing spans are spread across chunks about software restrictions and trademark use which embed as similar to license-grant queries but contain different annotations
- Audit Rights: 3/5 max recovery for all models — span 5/5 ("*More detailed quality audits may be performed by NCM personnel.*") is irrecoverable; its chunk embeds as quality standards content, not audit rights
- bge-large missed Anti-Assignment entirely in the question-form run (timed out on that query in one captured run; rank 1 in the short-query run)

---

## 7. Path to 95%+ Recall

Based on the experiments, the 96% ceiling (Ann@50) is real, not methodological. The remaining 4% requires:

| Approach | Expected gain | Effort |
|---|---|---|
| **Sentence-level indexing**: embed individual sentences as separate Qdrant points | +3–4% (the missing span gets its own vector) | High — re-ingest with sentence chunking |
| **BM25 keyword fallback**: for exact-match queries on known clause keywords | +1–2% | Medium |
| **Section detection + boosting**: identify section headers, boost chunks in audit-labeled sections | +1–2% | Medium |
| Multi-query decomposition | 0% (tested, no effect) | — |
| Smaller chunks (200-char) | 0% at top-50 (tested) | — |

---

## 8. How to Run the Pending Experiment (E6)

```bash
# Ingest remaining 49 docs into cuad_voyage_law2_hybrid_200 + build gold + run eval
# Estimated total time: ~4 hours (run in tmux or nohup)
bash tests/eval/run_experiment_voyage_law2_hybrid_200.sh 2>&1 | tee /tmp/e6_run.log
```

Once complete, compare all models:

```bash
python3 - <<'EOF'
import json, pathlib
runs = pathlib.Path("tests/eval/runs")
targets = [
    ("voyage-law-2 (500ch)",  "cuad_voyage_law2_hybrid_10"),
    ("voyage-law-2 (200ch)",  "cuad_voyage_law2_hybrid_200"),
    ("bge-large (500ch)",     "cuad_bgelarge_hybrid_50"),
    ("mpnet (500ch)",         "cuad_mpnet_hybrid_50"),
    ("MiniLM (500ch)",        "cuad_minilm_hybrid_50"),
]
print(f"{'Model':<25} {'cr@20':>6} {'cMRR':>6} {'sh@1':>6} {'sh@10':>6} {'sMRR':>6}")
print("─" * 70)
for label, prefix in targets:
    summaries = sorted(runs.glob(f"{prefix}*/summary.json"))
    if not summaries: print(f"{label:<25}  (no run)"); continue
    d = json.loads(summaries[-1].read_text())
    co, sh = d["contract_metrics"], d["span_hit"]
    print(f"{label:<25} {co['recall@20']:>6.3f} {co['mrr@20']:>6.3f} "
          f"{sh['hit@1']:>6.3f} {sh['hit@10']:>6.3f} {sh['mrr']:>6.3f}")
EOF
```

---

## 9. Server and Collection Reference

### Active eval servers (hybrid)

| Model | Collection | Port | Status |
|---|---|---|---|
| voyage-law-2 (500-char) | `cuad_voyage_law2_hybrid_10` | 8006 | Full corpus (50 docs) |
| voyage-law-2 (200-char) | `cuad_voyage_law2_hybrid_200` | 8007 | 1 doc (E6 pending) |
| bge-large-en-v1.5 | `cuad_bgelarge_hybrid_50` | 8013 | Full corpus (50 docs) |
| all-mpnet-base-v2 | `cuad_mpnet_hybrid_50` | 8012 | Full corpus (50 docs) |
| all-MiniLM-L6-v2 | `cuad_minilm_hybrid_50` | 8011 | Full corpus (50 docs) |

### Dense-only collections (no active server; used in E0a/E0b ablation)

| Model | Collection | Notes |
|---|---|---|
| all-MiniLM-L6-v2 | `cuad_sample_minilm_50` | Dense-only, 50 docs |
| all-mpnet-base-v2 | `cuad_sample_mpnet_50` | Dense-only, 50 docs |

Start all eval servers:
```bash
source .env.dev && bash tests/eval/start_eval_servers.sh
```
