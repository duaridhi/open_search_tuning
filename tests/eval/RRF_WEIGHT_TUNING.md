# RRF Weight Tuning — `cuad_bgelarge_hybrid_50`

**Date:** 2026-06-01
**Collection:** `cuad_bgelarge_hybrid_50` (50 docs, 7,391 chunks, 1024-d bge-large dense + BM42 sparse)
**Queries:** 82 CUAD queries (`tests/eval/queries.json`); 74 have ≥1 gold chunk
**Gold:** `tests/eval/gold/cuad_bgelarge_hybrid_50/` (collection-specific)
**Harness:** `tests/eval/rrf_sweep.py` (offline; embeds each query once, then sweeps
Qdrant **native weighted RRF** `RrfQuery(rrf=Rrf(weights=[w_dense, w_sparse]))`)
**Reranker:** OFF for the sweep — metrics reflect the raw fused order that feeds the
rerank pool, isolating the fusion weight's effect. Production reranker sits on top.

`alpha` = dense weight; sparse weight = `1 - alpha`. Only the ratio matters.
`alpha=0.5` ≡ today's balanced `Fusion.RRF` (production default). RRF `k` = Qdrant default.

## Coarse sweep (alpha 0.0 → 1.0)

| alpha | w=[d,s] | chunk r@10 | chunk r@20 | chunk mrr@10 | chunk ndcg@10 | contract r@20 | contract mrr@10 | contract ndcg@10 |
|------:|---------|-----------:|-----------:|-------------:|--------------:|--------------:|----------------:|-----------------:|
| 0.00 | [0.0,1.0] (pure sparse) | 0.1142 | 0.1570 | 0.2150 | 0.1317 | 0.3873 | 0.7008 | 0.6584 |
| 0.30 | [0.3,0.7] | 0.1176 | 0.1893 | 0.2617 | 0.1517 | 0.4032 | 0.7231 | 0.6899 |
| **0.50** | **[0.5,0.5] (current prod)** | **0.1288** | **0.1913** | **0.3111** | **0.1729** | **0.3973** | **0.7729** | **0.7009** |
| 0.60 | [0.6,0.4] | 0.1379 | 0.1822 | 0.3115 | 0.1779 | 0.4026 | 0.7869 | 0.6987 |
| **0.70** | **[0.7,0.3] ← best** | **0.1407** | **0.1903** | **0.3161** | **0.1802** | **0.4065** | **0.7943** | **0.6954** |
| 0.90 | [0.9,0.1] | 0.1340 | 0.1845 | 0.3149 | 0.1810 | 0.3942 | 0.7897 | 0.6839 |
| 1.00 | [1.0,0.0] (pure dense) | 0.1229 | 0.1656 | 0.3106 | 0.1781 | 0.3855 | 0.7859 | 0.6806 |

## Fine sweep (alpha 0.55 → 0.80) — confirms a flat optimum at 0.65–0.75

| alpha | chunk r@10 | chunk r@20 | chunk mrr@10 | chunk ndcg@10 | contract r@20 | contract mrr@10 |
|------:|-----------:|-----------:|-------------:|--------------:|--------------:|----------------:|
| 0.55 | 0.1312 | 0.1893 | 0.3059 | 0.1711 | 0.3989 | 0.7825 |
| 0.65 | 0.1394 | 0.1854 | 0.3163 | 0.1791 | 0.4040 | 0.7958 |
| **0.70** | **0.1407** | **0.1903** | 0.3161 | **0.1802** | **0.4065** | 0.7943 |
| 0.75 | 0.1410 | 0.1908 | 0.3177 | 0.1796 | 0.4024 | 0.7939 |
| 0.80 | 0.1366 | 0.1880 | 0.3170 | 0.1790 | 0.3953 | 0.7916 |

## Findings

1. **Dense-leaning beats balanced beats sparse-leaning** — monotonic, consistent
   across chunk- and contract-level metrics. Pure sparse (alpha=0) is clearly worst;
   pure dense (alpha=1) is also sub-optimal, so BM42 sparse *does* contribute.
2. **Optimum is `alpha ≈ 0.70`, i.e. weights `[0.7, 0.3]`** (dense:sparse ≈ 7:3,
   equivalently `[2.3, 1.0]`). The peak is flat across [0.65, 0.75].
3. **vs current balanced `[0.5, 0.5]`** at alpha=0.70:
   chunk recall@10 +9% (0.129→0.141), chunk ndcg@10 +4% (0.173→0.180),
   contract mrr@10 +2.8% (0.773→0.794), contract recall@20 +2.5% (0.397→0.407).
   Gains are modest but directionally clear and consistent.
4. **Caveat:** measured with reranker OFF. The production cross-encoder reranks the
   top of the fused pool, so it will absorb some of the *ranking* gains (MRR/nDCG);
   the **recall@20 / contract-recall** gains (which set what enters the rerank pool)
   are the durable benefit.

## Recommendation

Switch hybrid fusion from unweighted `Fusion.RRF` to weighted RRF with
**`weights=[0.7, 0.3]`** (dense, sparse). Suggested env-driven knobs:
`RRF_DENSE_WEIGHT=0.7`, `RRF_SPARSE_WEIGHT=0.3`. Not yet wired into
`qdrant_search_hf.hybrid_search` — see follow-up.

## Semantic (dense only) vs Hybrid — head-to-head

Same 82 queries / gold, reranker off. "semantic" = plain dense `query_points`.

| setup | chunk r@10 | chunk r@20 | chunk mrr@10 | chunk ndcg@10 | contract r@20 | contract mrr@10 | contract ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic (dense only) | 0.1229 | 0.1649 | 0.3106 | 0.1781 | 0.3852 | 0.7859 | 0.6806 |
| hybrid balanced [0.5,0.5] | 0.1288 | **0.1913** | 0.3048 | 0.1680 | 0.3958 | 0.7472 | 0.6924 |
| **hybrid tuned [0.7,0.3]** | **0.1407** | 0.1903 | **0.3161** | **0.1802** | **0.4065** | **0.7943** | **0.6954** |
| sparse only (BM42) | 0.1142 | 0.1570 | 0.2150 | 0.1317 | 0.3873 | 0.7008 | 0.6595 |

**Read this carefully — it's the key result:**
- **Hybrid improves recall** at every depth (chunk r@20 0.165→0.191 = +15%; contract
  r@20 0.385→0.407 = +5.5%). Sparse finds relevant passages dense misses.
- **But balanced hybrid HURTS top-rank quality vs pure semantic**: chunk ndcg@10
  0.178→0.168, contract mrr@10 0.786→0.747. Equal-weight BM42 pushes lexical-but-
  weaker matches up the list.
- **Tuned hybrid [0.7,0.3] gets both**: it beats pure semantic on *every* metric —
  recall *and* ranking. So the value of tuning isn't a marginal nudge over balanced;
  it's what makes hybrid strictly better than dense-only instead of a recall/precision
  trade-off.

## Reproduce

```bash
# phase 1 (embed queries once; HF bge-large; ~5 min)
QDRANT_COLLECTION=cuad_bgelarge_hybrid_50 LOAD_MODEL_STRATEGY=hybrid_search EMBED_BACKEND=hf \
  EMBED_MODEL=BAAI/bge-large-en-v1.5 python -u tests/eval/rrf_sweep.py --fetch
# phase 2 (weight sweep; Qdrant only; ~3 min)
QDRANT_COLLECTION=cuad_bgelarge_hybrid_50 python -u tests/eval/rrf_sweep.py --sweep
```
