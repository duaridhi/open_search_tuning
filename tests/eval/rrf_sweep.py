"""
rrf_sweep.py
────────────
Tune hybrid-search RRF weights on the cuad_bgelarge_hybrid_50 collection using
Qdrant's *native* weighted RRF:

    client.query_points(
        prefetch=[Prefetch(dense, using=""), Prefetch(sparse, using="sparse")],
        query=models.RrfQuery(rrf=models.Rrf(weights=[w_dense, w_sparse])),
    )

(see https://qdrant.tech/documentation/search/hybrid-queries/)

Because only the *ratio* of the two weights affects ordering, we sweep a single
alpha and set weights=[alpha, 1-alpha]:
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse,
    alpha=0.5 → equal weights ≡ today's balanced Fusion.RRF (production default).

The per-query dense + sparse query vectors are fixed, so we embed each query once
(phase 1, the only HF-API cost) and cache the raw vectors.  Phase 2 then re-issues
the weighted-RRF query to Qdrant for every weight in the grid — fast, no re-embed,
no cross-encoder reranker (so the metric isolates the fusion weight's effect on
retrieval quality, i.e. the order that feeds the rerank pool).

Phase 1 (embed+cache):  python -u tests/eval/rrf_sweep.py --fetch
Phase 2 (sweep):        python -u tests/eval/rrf_sweep.py --sweep
Both:                   python -u tests/eval/rrf_sweep.py --fetch --sweep

Env:
  QDRANT_COLLECTION  default cuad_bgelarge_hybrid_50
  EMBED_MODEL        default BAAI/bge-large-en-v1.5
  EMBED_BACKEND=hf   (HF Inference API; needs HF_TOKEN in .env.dev)
  LOAD_MODEL_STRATEGY=hybrid_search  (required; loads dense + BM42 sparse encoders)
  PREFETCH_N         candidates each prefetch pulls before fusion (default 120)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "cuad-demo-quadrant"))

COLLECTION = os.getenv("QDRANT_COLLECTION", "cuad_bgelarge_hybrid_50")
PREFETCH_N = int(os.getenv("PREFETCH_N", "120"))
TOP_K = 20
VEC_CACHE = EVAL_DIR / "rrf_cache" / f"{COLLECTION}_vectors.json"

QUERIES_PATH = EVAL_DIR / "queries.json"
# Gold is collection-specific (chunk ids depend on the ingest/chunking of THIS
# collection). The per-collection gold lives under tests/eval/gold/<collection>/;
# override with GOLD_DIR. (The top-level tests/eval/gold.json belongs to a
# different collection and must NOT be used here — its chunk ids won't match.)
GOLD_DIR = Path(os.getenv("GOLD_DIR", EVAL_DIR / "gold" / COLLECTION))
GOLD_PATH = GOLD_DIR / "gold.json"
GOLD_CONTRACTS_PATH = GOLD_DIR / "gold_contracts.json"


# ── metrics (mirror run_eval.py) ─────────────────────────────────────────────
def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & gold) / len(gold) if gold else None


def precision_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & gold) / k if gold and k else None


def mrr(retrieved, gold, k):
    if not gold:
        return None
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, gold, k):
    if not gold:
        return None
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(retrieved[:k]) if it in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(retrieved, gold):
    return {
        "recall@5": recall_at_k(retrieved, gold, 5),
        "recall@10": recall_at_k(retrieved, gold, 10),
        "recall@20": recall_at_k(retrieved, gold, 20),
        "precision@5": precision_at_k(retrieved, gold, 5),
        "precision@10": precision_at_k(retrieved, gold, 10),
        "mrr@10": mrr(retrieved, gold, 10),
        "mrr@20": mrr(retrieved, gold, 20),
        "ndcg@10": ndcg_at_k(retrieved, gold, 10),
    }


METRIC_KEYS = ["recall@5", "recall@10", "recall@20", "precision@5",
               "precision@10", "mrr@10", "mrr@20", "ndcg@10"]


def macro(rows, group, k):
    vals = [r[group][k] for r in rows if r[group].get(k) is not None]
    return statistics.mean(vals) if vals else None


# ── phase 1: embed each query once, cache raw dense + sparse vectors ──────────
def _embed_retry(embed_query, text, tries=6):
    """HF serverless throws transient 500/429s; retry with exponential backoff."""
    for attempt in range(tries):
        try:
            return embed_query(text)
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = min(30, 2 ** attempt)
            print(f"    embed failed ({type(e).__name__}); retry {attempt+1}/{tries} in {wait}s", flush=True)
            time.sleep(wait)


def fetch():
    from qdrant_search_hf import embed_query, _sparse_encoder

    if _sparse_encoder is None:
        sys.exit("Sparse model not loaded; run with LOAD_MODEL_STRATEGY=hybrid_search (or sparse_search).")

    queries = json.loads(QUERIES_PATH.read_text())
    # Resume: keep any vectors already cached from a prior (crashed) run.
    VEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(VEC_CACHE.read_text()) if VEC_CACHE.exists() else {}
    if cache:
        print(f"Resuming: {len(cache)} queries already cached", flush=True)
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        qid = q.get("id") or q["q"]
        if qid in cache:
            continue
        t0 = time.perf_counter()
        dvec = _embed_retry(embed_query, q["q"])
        sp = list(_sparse_encoder.embed([q["q"]]))[0]
        cache[qid] = {
            "dense": list(dvec),
            "sparse": {"indices": sp.indices.tolist(), "values": sp.values.tolist()},
        }
        VEC_CACHE.write_text(json.dumps(cache))  # incremental save → crash-safe
        print(f"[{i+1}/{len(queries)}] {qid}  ({time.perf_counter()-t0:.2f}s)", flush=True)

    print(f"\nCached {len(cache)} query vectors → {VEC_CACHE} "
          f"({time.perf_counter()-t_start:.1f}s)", flush=True)


# ── phase 2: server-side weighted-RRF sweep ──────────────────────────────────
def sweep(alphas, ks):
    from qdrant_cluster_connect import get_qdrant_client
    from qdrant_client import models

    client = get_qdrant_client()
    print(f"Gold dir: {GOLD_DIR}", flush=True)
    vecs = json.loads(VEC_CACHE.read_text())
    gold = json.loads(GOLD_PATH.read_text())
    gold_contracts = json.loads(GOLD_CONTRACTS_PATH.read_text())
    queries = json.loads(QUERIES_PATH.read_text())
    qids = [q.get("id") or q["q"] for q in queries]

    configs = []
    for k in ks:
        for alpha in alphas:
            w = [round(alpha, 4), round(1.0 - alpha, 4)]
            rows = []
            t0 = time.perf_counter()
            for qid in qids:
                v = vecs[qid]
                sparse = models.SparseVector(
                    indices=v["sparse"]["indices"], values=v["sparse"]["values"]
                )
                rrf = models.Rrf(weights=w) if k is None else models.Rrf(weights=w, k=k)
                res = client.query_points(
                    collection_name=COLLECTION,
                    prefetch=[
                        models.Prefetch(query=v["dense"], using="", limit=PREFETCH_N),
                        models.Prefetch(query=sparse, using="sparse", limit=PREFETCH_N),
                    ],
                    query=models.RrfQuery(rrf=rrf),
                    limit=TOP_K,
                    with_payload=["doc_id", "title"],
                ).points
                ids = [(p.payload or {}).get("doc_id") for p in res]
                titles = [(p.payload or {}).get("title", "") for p in res]
                rows.append({
                    "chunk": compute_metrics(ids, set(gold.get(qid, []))),
                    "contract": compute_metrics(
                        titles, set((gold_contracts.get(qid) or {}).get("relevant_titles") or [])),
                })
            configs.append({
                "alpha": alpha, "weights": w, "k": k,
                "chunk": {m: macro(rows, "chunk", m) for m in METRIC_KEYS},
                "contract": {m: macro(rows, "contract", m) for m in METRIC_KEYS},
            })
            print(f"k={k} alpha={alpha:.2f} weights={w} "
                  f"chunk_r@20={configs[-1]['chunk']['recall@20']:.4f} "
                  f"contract_r@10={configs[-1]['contract']['recall@10']:.4f} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return configs


def compare():
    """Semantic (pure dense) vs hybrid (dense+sparse weighted RRF), same queries/gold.

    'semantic' issues a plain dense query_points(using="") — exactly what
    semantic_search() does — so this is a faithful dense-only vs hybrid comparison
    (reranker off on both sides)."""
    from qdrant_cluster_connect import get_qdrant_client
    from qdrant_client import models

    client = get_qdrant_client()
    print(f"Gold dir: {GOLD_DIR}", flush=True)
    vecs = json.loads(VEC_CACHE.read_text())
    gold = json.loads(GOLD_PATH.read_text())
    gold_contracts = json.loads(GOLD_CONTRACTS_PATH.read_text())
    queries = json.loads(QUERIES_PATH.read_text())
    qids = [q.get("id") or q["q"] for q in queries]

    # (label, mode, weights) — mode "dense" = pure semantic, "rrf" = hybrid fusion
    setups = [
        ("semantic (dense only)", "dense", None),
        ("hybrid balanced [0.5,0.5]", "rrf", [0.5, 0.5]),
        ("hybrid tuned [0.7,0.3]", "rrf", [0.7, 0.3]),
        ("sparse only (BM42)", "rrf", [0.0, 1.0]),
    ]
    results = []
    for label, mode, w in setups:
        rows = []
        t0 = time.perf_counter()
        for qid in qids:
            v = vecs[qid]
            if mode == "dense":
                pts = client.query_points(
                    collection_name=COLLECTION, query=v["dense"], using="",
                    limit=TOP_K, with_payload=["doc_id", "title"]).points
            else:
                sparse = models.SparseVector(
                    indices=v["sparse"]["indices"], values=v["sparse"]["values"])
                pts = client.query_points(
                    collection_name=COLLECTION,
                    prefetch=[
                        models.Prefetch(query=v["dense"], using="", limit=PREFETCH_N),
                        models.Prefetch(query=sparse, using="sparse", limit=PREFETCH_N),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(weights=w)),
                    limit=TOP_K, with_payload=["doc_id", "title"]).points
            ids = [(p.payload or {}).get("doc_id") for p in pts]
            titles = [(p.payload or {}).get("title", "") for p in pts]
            rows.append({
                "chunk": compute_metrics(ids, set(gold.get(qid, []))),
                "contract": compute_metrics(
                    titles, set((gold_contracts.get(qid) or {}).get("relevant_titles") or [])),
            })
        results.append({
            "label": label,
            "chunk": {m: macro(rows, "chunk", m) for m in METRIC_KEYS},
            "contract": {m: macro(rows, "contract", m) for m in METRIC_KEYS},
        })
        print(f"  ran {label}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    cols = ["recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"]
    for level in ("chunk", "contract"):
        print(f"\n═══ {level.upper()}-LEVEL ═══")
        hdr = f"{'setup':<28}| " + " ".join(f"{c:>9}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for r in results:
            vals = " ".join(f"{(r[level][c] or 0):>9.4f}" for c in cols)
            print(f"{r['label']:<28}| {vals}")
    (EVAL_DIR / "rrf_cache" / "compare_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {EVAL_DIR / 'rrf_cache' / 'compare_results.json'}", flush=True)


def print_table(configs, ks):
    cols = ["recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"]
    for k in ks:
        print(f"\n═══ Qdrant weighted RRF  (k={k if k is not None else 'default(2)'}) "
              f" weights=[alpha, 1-alpha] ═══")
        print(f"{'':>20}CHUNK-LEVEL{'':>34}CONTRACT-LEVEL")
        hdr = "alpha  w=[d,s]   | " + " ".join(f"{c:>9}" for c in cols) + "  |" + \
              " ".join(f"{c:>9}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for cfg in [c for c in configs if c["k"] == k]:
            ch = " ".join(f"{(cfg['chunk'][c] or 0):>9.4f}" for c in cols)
            co = " ".join(f"{(cfg['contract'][c] or 0):>9.4f}" for c in cols)
            tag = "  <- balanced (current prod)" if abs(cfg["alpha"] - 0.5) < 1e-9 else ""
            wd, ws = cfg["weights"]
            print(f"{cfg['alpha']:>5.2f} [{wd:.2f},{ws:.2f}] | {ch}  |{co}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="semantic (dense) vs hybrid (RRF) head-to-head")
    ap.add_argument("--alphas", default="", help="comma list; default 0.0..1.0 step 0.1")
    ap.add_argument("--ks", default="default", help="comma list of RRF k; 'default' = Qdrant default")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "rrf_cache" / "sweep_results.json")
    args = ap.parse_args()

    if args.fetch:
        fetch()
    if args.compare:
        if not VEC_CACHE.exists():
            sys.exit(f"No vector cache at {VEC_CACHE}; run with --fetch first.")
        compare()
        return
    if args.sweep or not args.fetch:
        if not VEC_CACHE.exists():
            sys.exit(f"No vector cache at {VEC_CACHE}; run with --fetch first.")
        alphas = ([round(i / 10, 2) for i in range(11)]
                  if not args.alphas else [float(x) for x in args.alphas.split(",")])
        ks = [None if x == "default" else int(x) for x in args.ks.split(",")]
        configs = sweep(alphas, ks)
        print_table(configs, ks)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(configs, indent=2))
        print(f"\nFull results → {args.out}", flush=True)


if __name__ == "__main__":
    main()
