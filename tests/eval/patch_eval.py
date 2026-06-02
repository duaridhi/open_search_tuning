"""
patch_eval.py — Re-run failed queries from an existing eval run and recalculate summary.

Reads the search.json from a prior run, finds entries where error != null,
re-queries the live server for each, patches the results in place, and rewrites
both search.json and summary.json.  All successful entries are preserved unchanged.

Usage:
    python tests/eval/patch_eval.py --run-dir tests/eval/runs/<run>/
    python tests/eval/patch_eval.py --run-dir tests/eval/runs/<run>/ --base-url http://localhost:8006
    python tests/eval/patch_eval.py --run-dir tests/eval/runs/<run>/ --query-sleep 21 --dry-run

Options:
    --run-dir       Path to an existing run directory (required).
    --base-url      Server base URL (default: http://localhost:8006).
    --strategy      Search strategy (default: auto-detected from summary, else hybrid_search).
    --query-sleep   Seconds to sleep between queries (default: 21 for VoyageAI free tier).
    --dry-run       Print which queries would be re-run without touching any files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("patch_eval")

EVAL_DIR = Path(__file__).resolve().parent
TOP_K = 20
LATENCY_REPS = 5
SEARCH_TIMEOUT_S = 180.0
METRIC_KEYS = [
    "recall@5", "recall@10", "recall@20",
    "precision@5", "precision@10",
    "mrr@20", "ndcg@10",
]

_CURLY_TABLE = str.maketrans({0x201C: chr(34), 0x201D: chr(34), 0x2018: chr(39), 0x2019: chr(39)})
MIN_SPAN_TEXT_LEN = 20


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.translate(_CURLY_TABLE)).strip()


# ── Metric helpers (mirrors run_eval.py) ────────────────────────────────────

def recall_at_k(retrieved: list, gold: set, k: int) -> float | None:
    if not gold:
        return None
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: list, gold: set, k: int) -> float | None:
    if not gold or k == 0:
        return None
    return len(set(retrieved[:k]) & gold) / k


def mrr(retrieved: list, gold: set, k: int) -> float | None:
    if not gold:
        return None
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list, gold: set, k: int) -> float | None:
    if not gold:
        return None
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[idx]


def compute_metrics(retrieved: list, gold: set) -> dict:
    return {k: f(retrieved, gold) for k, f in [
        ("recall@5",    lambda r, g: recall_at_k(r, g, 5)),
        ("recall@10",   lambda r, g: recall_at_k(r, g, 10)),
        ("recall@20",   lambda r, g: recall_at_k(r, g, 20)),
        ("precision@5", lambda r, g: precision_at_k(r, g, 5)),
        ("precision@10",lambda r, g: precision_at_k(r, g, 10)),
        ("mrr@20",      lambda r, g: mrr(r, g, 20)),
        ("ndcg@10",     lambda r, g: ndcg_at_k(r, g, 10)),
    ]}


def _compute_span_hit(results: list[dict], spans_by_doc: dict[str, list[dict]]) -> dict:
    if not spans_by_doc:
        return {"hit@1": None, "hit@5": None, "hit@10": None, "first_hit_rank": None}
    first_hit_rank = None
    for rank, res in enumerate(results, start=1):
        title = res.get("title") or ""
        chunk_norm = _norm(res.get("text") or "")
        for s in spans_by_doc.get(title, []):
            if len(s.get("answer_text", "")) < MIN_SPAN_TEXT_LEN:
                continue
            if _norm(s["answer_text"]) in chunk_norm:
                if first_hit_rank is None:
                    first_hit_rank = rank
                break
    return {
        "hit@1":  first_hit_rank is not None and first_hit_rank <= 1,
        "hit@5":  first_hit_rank is not None and first_hit_rank <= 5,
        "hit@10": first_hit_rank is not None and first_hit_rank <= 10,
        "first_hit_rank": first_hit_rank,
    }


def macro_avg(per_query: list[dict], metrics_key: str, k: str) -> float | None:
    vals = [
        pq[metrics_key].get(k)
        for pq in per_query
        if pq.get(metrics_key) and pq[metrics_key].get(k) is not None
    ]
    return statistics.mean(vals) if vals else None


def summarize(per_query: list[dict], gold_contracts: dict) -> dict:
    summary: dict[str, Any] = {
        "n_queries": len(per_query),
        "chunk_metrics":    {k: macro_avg(per_query, "chunk_metrics", k)    for k in METRIC_KEYS},
        "contract_metrics": {k: macro_avg(per_query, "contract_metrics", k) for k in METRIC_KEYS},
    }
    for k in ("hit@1", "hit@5", "hit@10"):
        vals = [
            pq["span_hit"][k]
            for pq in per_query
            if pq.get("span_hit") and pq["span_hit"].get(k) is not None
        ]
        summary.setdefault("span_hit", {})[k] = sum(vals) / len(vals) if vals else None
    mrr_ranks = [
        pq["span_hit"]["first_hit_rank"]
        for pq in per_query
        if pq.get("span_hit") and pq["span_hit"].get("first_hit_rank") is not None
    ]
    summary.setdefault("span_hit", {})["mrr"] = (
        statistics.mean(1.0 / r for r in mrr_ranks) if mrr_ranks else None
    )
    lat_p50 = [pq["latency_s"]["p50"] for pq in per_query if pq["latency_s"]["reps"]]
    lat_p95 = [pq["latency_s"]["p95"] for pq in per_query if pq["latency_s"]["reps"]]
    summary["latency_s"] = {
        "p50_median_across_queries": percentile(lat_p50, 50) if lat_p50 else None,
        "p95_median_across_queries": percentile(lat_p95, 50) if lat_p95 else None,
        "p95_max_across_queries":    max(lat_p95) if lat_p95 else None,
    }
    summary["zero_result_queries"] = [pq["id"] for pq in per_query if not pq["retrieved_ids"]]
    return summary


# ── HTTP helper ──────────────────────────────────────────────────────────────

def call_search(
    client: httpx.Client, base_url: str, q: str, doc: str | None, strategy: str,
) -> tuple[list, list, list[dict], float]:
    params: dict[str, Any] = {"q": q, "top_k": TOP_K, "strategy": strategy}
    if doc:
        params["document_name"] = doc
    t0 = time.perf_counter()
    r = client.get(f"{base_url}/search", params=params, timeout=SEARCH_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results") or payload.get("hits") or []
    ids     = [res.get("id") for res in results if res.get("id") is not None]
    titles  = [(res.get("title") or "") for res in results]
    return ids, titles, results, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gold-dir", type=Path, default=None,
                    help="Directory containing gold.json / gold_contracts.json / gold_spans.json. "
                         "Defaults to <run-dir>, then tests/eval/gold/<collection-name>/, "
                         "then tests/eval/.")
    ap.add_argument("--base-url", default="http://localhost:8006")
    ap.add_argument("--strategy", default=None,
                    help="Search strategy (default: auto-detect from existing summary)")
    ap.add_argument("--query-sleep", type=float, default=21.0, metavar="SECONDS")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    search_path  = run_dir / "search.json"
    summary_path = run_dir / "summary.json"

    if not search_path.exists():
        log.error("search.json not found in %s", run_dir)
        return 2

    per_query: list[dict] = json.loads(search_path.read_text())
    failed = [pq for pq in per_query if pq.get("error")]
    log.info("Run dir      : %s", run_dir)
    log.info("Total queries: %d  |  failed: %d  |  ok: %d",
             len(per_query), len(failed), len(per_query) - len(failed))

    if not failed:
        log.info("No failed queries — nothing to patch.")
        return 0

    if args.dry_run:
        log.info("DRY RUN — would re-run:")
        for pq in failed:
            log.info("  %s  error=%s", pq["id"], (pq.get("error") or "")[:80])
        return 0

    # Auto-detect strategy from existing summary if not specified
    strategy = args.strategy
    if not strategy and summary_path.exists():
        existing_summary = json.loads(summary_path.read_text())
        strategy = existing_summary.get("strategy", "hybrid_search")
    strategy = strategy or "hybrid_search"
    log.info("Strategy     : %s", strategy)

    # Locate gold directory — explicit flag, then infer from collection name in run dir name
    eval_dir = Path(__file__).resolve().parent
    if args.gold_dir:
        gold_candidates = [args.gold_dir]
    else:
        # run_dir name format: <collection>_<model>_<timestamp>
        # collection name is the first segment before the model slug
        parts = run_dir.name.split("_")
        inferred_collection = "_".join(parts[:3])  # e.g. cuad_voyage_law2
        gold_candidates = [
            run_dir,
            eval_dir / "gold" / inferred_collection,
            eval_dir / "gold" / run_dir.name,
            eval_dir,
        ]

    def _load_gold(filename: str) -> dict:
        for d in gold_candidates:
            p = d / filename
            if p.exists():
                log.info("Gold %s: %s", filename, p)
                return json.loads(p.read_text())
        log.warning("Gold file not found: %s (searched %s)", filename,
                    [str(d) for d in gold_candidates])
        return {}

    gold_chunks    = _load_gold("gold.json")
    gold_contracts = _load_gold("gold_contracts.json")
    gold_spans     = _load_gold("gold_spans.json")

    log.info("Pinging %s ...", args.base_url)
    with httpx.Client() as client:
        try:
            client.get(f"{args.base_url}/health", timeout=10.0).raise_for_status()
        except Exception as e:
            log.error("Cannot reach %s/health: %s", args.base_url, e)
            return 3

        # Build lookup by id for fast patching
        by_id = {pq["id"]: pq for pq in per_query}
        patched = 0

        for i, pq in enumerate(failed):
            qid = pq["id"]
            q_text = pq["q"]
            doc = pq.get("doc")

            log.info("[%d/%d] Re-running %s", i + 1, len(failed), qid)
            latencies: list[float] = []
            ids: list = []
            titles: list = []
            results: list = []
            last_err: str | None = None

            for rep in range(LATENCY_REPS):
                try:
                    ids, titles, results, el = call_search(client, args.base_url, q_text, doc, strategy)
                    latencies.append(el)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    log.warning("  rep %d failed: %s", rep, last_err)
                    break

            gold_chunk_set = set(gold_chunks.get(qid, []))
            gold_title_set = set((gold_contracts.get(qid) or {}).get("relevant_titles") or [])
            gold_spans_for_q = (gold_spans.get(qid) or {}).get("spans_by_document") or {}

            by_id[qid].update({
                "retrieved_ids":    ids,
                "retrieved_titles": titles,
                "latency_s": {
                    "p50":  percentile(latencies, 50),
                    "p95":  percentile(latencies, 95),
                    "reps": len(latencies),
                },
                "chunk_metrics":    compute_metrics(ids, gold_chunk_set),
                "contract_metrics": compute_metrics(titles, gold_title_set),
                "span_hit":         _compute_span_hit(results, gold_spans_for_q),
                "top_highlights": [
                    {
                        "id": r.get("id"),
                        "title": r.get("title"),
                        "highlight": (r.get("highlight") or r.get("highlights") or [None])[:1],
                    }
                    for r in results[:5]
                ],
                "error": last_err,
            })
            if not last_err:
                patched += 1

            if args.query_sleep > 0 and i < len(failed) - 1:
                time.sleep(args.query_sleep)

    log.info("Patched %d/%d failed queries successfully.", patched, len(failed))

    # Back up original files
    search_path.rename(run_dir / "search.json.bak")
    if summary_path.exists():
        summary_path.rename(run_dir / "summary.json.bak")
    log.info("Originals backed up as search.json.bak / summary.json.bak")

    # Write updated files
    updated = list(by_id.values())
    search_path.write_text(json.dumps(updated, indent=2))
    summary = summarize(updated, gold_contracts)
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Updated search.json and summary.json written to %s", run_dir)
    still_failed = [pq["id"] for pq in updated if pq.get("error")]
    if still_failed:
        log.warning("%d queries still have errors: %s", len(still_failed), still_failed)
    else:
        log.info("All queries now have results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
