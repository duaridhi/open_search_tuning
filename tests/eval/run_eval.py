"""
run_eval.py
───────────
Run the canonical query set against a live /search (and optionally /chat),
score binary retrieval metrics against tests/eval/gold.json, and dump
per-query JSON plus a summary.json into tests/eval/runs/<timestamp>/.

Metrics (binary, macro-averaged):
  Recall@{5,10,20}
  Precision@{5,10}
  MRR@20
  nDCG@10
Latency: p50/p95 from 5 repeats of /search per query.
Optional --chat: deterministic Source-Title-Coverage check.

Usage:
  python tests/eval/run_eval.py                              # 82 CUAD queries
  python tests/eval/run_eval.py --queries tests/eval/queries_smoke.json
  python tests/eval/run_eval.py --chat                       # also hit /chat
  python tests/eval/run_eval.py --base-url http://localhost:8000
  python tests/eval/run_eval.py --out tests/eval/runs/manual/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_eval")

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = EVAL_DIR / "queries.json"
GOLD_PATH = EVAL_DIR / "gold.json"
RUNS_DIR = EVAL_DIR / "runs"

TOP_K = 20
LATENCY_REPS = 5
SEARCH_TIMEOUT_S = 120.0
CHAT_TIMEOUT_S = 180.0


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
    for i, pid in enumerate(retrieved[:k]):
        if pid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list, gold: set, k: int) -> float | None:
    if not gold:
        return None
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k]):
        if pid in gold:
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


def call_search(client: httpx.Client, base_url: str, q: str, doc: str | None) -> tuple[list, list[dict], float]:
    params: dict[str, Any] = {"q": q, "top_k": TOP_K, "strategy": "semantic_search"}
    if doc:
        params["document_name"] = doc
    t0 = time.perf_counter()
    r = client.get(f"{base_url}/search", params=params, timeout=SEARCH_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results") or payload.get("hits") or []
    ids = [res.get("id") for res in results if res.get("id") is not None]
    return ids, results, elapsed


def call_chat(client: httpx.Client, base_url: str, q: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/chat", json={"query": q, "top_k": 10}, timeout=CHAT_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    return {"latency_s": elapsed, "answer": body.get("answer", ""), "sources": body.get("sources", [])}


def evaluate_query(
    client: httpx.Client,
    base_url: str,
    q: dict,
    gold_ids: list,
    do_chat: bool,
) -> dict:
    gold_set = set(gold_ids)
    latencies: list[float] = []
    ids: list = []
    results: list = []
    last_err: str | None = None
    for i in range(LATENCY_REPS):
        try:
            ids, results, el = call_search(client, base_url, q["q"], q.get("doc"))
            latencies.append(el)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning("search failed [%s rep %d]: %s", q["id"], i, last_err)
            break

    out: dict[str, Any] = {
        "id": q["id"],
        "q": q["q"],
        "doc": q.get("doc"),
        "category": q.get("category"),
        "form": q.get("form"),
        "gold_count": len(gold_set),
        "retrieved_ids": ids,
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "reps": len(latencies),
        },
        "metrics": {
            "recall@5": recall_at_k(ids, gold_set, 5),
            "recall@10": recall_at_k(ids, gold_set, 10),
            "recall@20": recall_at_k(ids, gold_set, 20),
            "precision@5": precision_at_k(ids, gold_set, 5),
            "precision@10": precision_at_k(ids, gold_set, 10),
            "mrr@20": mrr(ids, gold_set, 20),
            "ndcg@10": ndcg_at_k(ids, gold_set, 10),
        },
        "top_highlights": [
            {"id": r.get("id"), "title": r.get("title"), "highlight": (r.get("highlight") or r.get("highlights") or [None])[:1]}
            for r in results[:5]
        ],
        "error": last_err,
    }

    if do_chat:
        try:
            chat = call_chat(client, base_url, q["q"])
            # Source-Title-Coverage: at least one cited title appears among
            # gold-bearing titles (we approximate "gold-bearing" by titles of
            # any gold point; the actual mapping is computed at summary time).
            out["chat"] = chat
        except Exception as e:
            out["chat"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def macro_avg(per_query: list[dict], key: str) -> float | None:
    vals = [pq["metrics"].get(key) for pq in per_query if pq["metrics"].get(key) is not None]
    return statistics.mean(vals) if vals else None


def summarize(per_query: list[dict], gold_titles_by_qid: dict[str, set[str]]) -> dict:
    keys = ["recall@5", "recall@10", "recall@20", "precision@5", "precision@10", "mrr@20", "ndcg@10"]
    summary: dict[str, Any] = {"n_queries": len(per_query), "macro": {k: macro_avg(per_query, k) for k in keys}}
    lat_p50 = [pq["latency_s"]["p50"] for pq in per_query if pq["latency_s"]["reps"]]
    lat_p95 = [pq["latency_s"]["p95"] for pq in per_query if pq["latency_s"]["reps"]]
    summary["latency_s"] = {
        "p50_median_across_queries": percentile(lat_p50, 50) if lat_p50 else None,
        "p95_median_across_queries": percentile(lat_p95, 50) if lat_p95 else None,
        "p95_max_across_queries": max(lat_p95) if lat_p95 else None,
    }
    summary["zero_result_queries"] = [pq["id"] for pq in per_query if not pq["retrieved_ids"]]

    # /chat source-title coverage if present.
    chat_rows = [pq for pq in per_query if "chat" in pq and isinstance(pq.get("chat"), dict) and "answer" in pq["chat"]]
    if chat_rows:
        covered = 0
        cannot = 0
        for pq in chat_rows:
            titles_in_chat = {s.get("title") for s in pq["chat"].get("sources") or [] if s.get("title")}
            gold_titles = gold_titles_by_qid.get(pq["id"], set())
            if gold_titles and titles_in_chat & gold_titles:
                covered += 1
            ans = (pq["chat"].get("answer") or "").lower()
            if "cannot determine" in ans or "i cannot" in ans:
                cannot += 1
        summary["chat"] = {
            "n": len(chat_rows),
            "source_title_coverage": covered / len(chat_rows) if chat_rows else None,
            "cannot_determine_count": cannot,
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    if not args.queries.exists():
        log.error("queries file not found: %s", args.queries)
        return 2
    queries = json.loads(args.queries.read_text())
    log.info("Loaded %d queries from %s", len(queries), args.queries)

    gold: dict[str, list] = {}
    gold_titles_by_qid: dict[str, set[str]] = {}
    if GOLD_PATH.exists():
        gold = json.loads(GOLD_PATH.read_text())
        log.info("Loaded gold for %d query ids", len(gold))
    else:
        log.warning("No gold.json yet; metrics will all be null. Run build_gold.py first.")

    out_dir = args.out or RUNS_DIR / dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing to %s", out_dir)

    # Sanity ping.
    with httpx.Client() as client:
        try:
            r = client.get(f"{args.base_url}/health", timeout=10.0)
            r.raise_for_status()
        except Exception as e:
            log.error("Cannot reach %s/health: %s", args.base_url, e)
            return 3

        per_query: list[dict] = []
        for i, q in enumerate(queries):
            qid = q.get("id") or q.get("q")
            gold_ids = gold.get(qid, [])
            log.info("[%d/%d] %s (gold=%d)", i + 1, len(queries), qid, len(gold_ids))
            row = evaluate_query(client, args.base_url, q, gold_ids, do_chat=args.chat)
            per_query.append(row)

    # Build qid → gold-bearing titles map from per_query retrieved_ids would
    # be wrong; instead derive from the gold file + Qdrant. For now, we
    # approximate via titles seen in top-K results that overlap gold ids.
    # (Cheap; only used as a sanity input to the chat coverage metric.)
    for pq in per_query:
        gids = set(gold.get(pq["id"], []))
        titles = set()
        for r in pq.get("top_highlights", []):
            if r.get("id") in gids and r.get("title"):
                titles.add(r["title"])
        gold_titles_by_qid[pq["id"]] = titles

    (out_dir / "search.json").write_text(json.dumps(per_query, indent=2))
    summary = summarize(per_query, gold_titles_by_qid)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Summary written to %s/summary.json", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
