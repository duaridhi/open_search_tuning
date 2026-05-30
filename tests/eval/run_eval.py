"""
run_eval.py
───────────
Run the canonical query set against a live /search (and optionally /chat),
score binary retrieval metrics against tests/eval/gold.json (chunk-level)
and tests/eval/gold_contracts.json (contract-level), and dump per-query
JSON plus a summary.json into tests/eval/runs/<timestamp>/.

Metrics (binary, macro-averaged):
  chunk_metrics:
    Recall@{5,10,20}, Precision@{5,10}, MRR@20, nDCG@10
  contract_metrics:
    Recall@{5,10,20}, Precision@{5,10}, MRR@20, nDCG@10
Latency: p50/p95 from 5 repeats of /search per query.
Optional --chat: deterministic Source-Title-Coverage check.

Usage:
  python tests/eval/run_eval.py
  python tests/eval/run_eval.py --queries tests/eval/queries_smoke.json
  python tests/eval/run_eval.py --chat
  python tests/eval/run_eval.py --base-url http://localhost:8000
  python tests/eval/run_eval.py --out tests/eval/runs/manual/
"""

from __future__ import annotations

import argparse
import datetime as dt
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
log = logging.getLogger("run_eval")

# Normaliser for text-based span matching (mirrors build_gold.py).
# CUAD spans reference TXT-derived char positions; Qdrant stores PDF-extracted
# positions.  The two texts differ in: (1) extra blank lines around section
# headers / page numbers (cumulative drift); (2) intra-paragraph \n vs space;
# (3) Unicode smart quotes vs ASCII.  Normalising before substring search fixes
# all three without touching the ingest pipeline.
_CURLY_TABLE = str.maketrans({0x201C: chr(34), 0x201D: chr(34), 0x2018: chr(39), 0x2019: chr(39)})
MIN_SPAN_TEXT_LEN = 20


def _norm(s: str) -> str:
    """Collapse whitespace and fold curly quotes → ASCII."""
    return re.sub(r'\s+', ' ', s.translate(_CURLY_TABLE)).strip()

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = EVAL_DIR / "queries.json"
RUNS_DIR = EVAL_DIR / "runs"

TOP_K = 20
LATENCY_REPS = 5
SEARCH_TIMEOUT_S = 120.0
CHAT_TIMEOUT_S = 180.0

METRIC_KEYS = [
    "recall@5",
    "recall@10",
    "recall@20",
    "precision@5",
    "precision@10",
    "mrr@20",
    "ndcg@10",
]


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
    return {
        "recall@5": recall_at_k(retrieved, gold, 5),
        "recall@10": recall_at_k(retrieved, gold, 10),
        "recall@20": recall_at_k(retrieved, gold, 20),
        "precision@5": precision_at_k(retrieved, gold, 5),
        "precision@10": precision_at_k(retrieved, gold, 10),
        "mrr@20": mrr(retrieved, gold, 20),
        "ndcg@10": ndcg_at_k(retrieved, gold, 10),
    }


def call_search(
    client: httpx.Client, base_url: str, q: str, doc: str | None, strategy: str = "semantic_search"
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
    ids = [res.get("id") for res in results if res.get("id") is not None]
    titles = [(res.get("title") or "") for res in results]
    return ids, titles, results, elapsed


def call_chat(client: httpx.Client, base_url: str, q: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/chat", json={"query": q, "top_k": 10}, timeout=CHAT_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    return {"latency_s": elapsed, "answer": body.get("answer", ""), "sources": body.get("sources", [])}


def _compute_span_hit(results: list[dict], spans_by_doc: dict[str, list[dict]]) -> dict:
    """For each k in {1,5,10}, check if any result in top-k contains a gold answer_text.

    Uses normalised text substring matching instead of char-offset overlap.
    CUAD gold char positions reference TXT-derived text; Qdrant stores PDF-extracted
    positions.  The two diverge by up to tens of chars per page due to differing
    blank-line counts around section headers and page numbers.  Text matching is
    source-agnostic and survives all three causes of drift identified in
    tests/eval/cuad_txt_vs_pdf_extract_analysis/.

    Returns {hit@1: bool|None, hit@5: bool|None, hit@10: bool|None,
             first_hit_rank: int|None}.
    None when gold spans are empty (query skipped from span-hit metrics).
    """
    if not spans_by_doc:
        return {"hit@1": None, "hit@5": None, "hit@10": None, "first_hit_rank": None}

    first_hit_rank = None
    for rank, res in enumerate(results, start=1):
        title = res.get("title") or ""
        chunk_norm = _norm(res.get("text") or "")
        doc_spans = spans_by_doc.get(title, [])
        for s in doc_spans:
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


def evaluate_query(
    client: httpx.Client,
    base_url: str,
    q: dict,
    gold_chunk_ids: list,
    gold_titles: list,
    gold_spans_for_query: dict[str, list[dict]],
    do_chat: bool,
    strategy: str = "semantic_search",
) -> dict:
    gold_chunk_set = set(gold_chunk_ids)
    gold_title_set = set(gold_titles)
    latencies: list[float] = []
    ids: list = []
    titles: list = []
    results: list = []
    last_err: str | None = None
    for i in range(LATENCY_REPS):
        try:
            ids, titles, results, el = call_search(client, base_url, q["q"], q.get("doc"), strategy=strategy)
            latencies.append(el)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning("search failed [%s rep %d]: %s", q.get("id") or q["q"], i, last_err)
            break

    out: dict[str, Any] = {
        "id": q.get("id") or q["q"],
        "q": q["q"],
        "doc": q.get("doc"),
        "category": q.get("category"),
        "form": q.get("form"),
        "gold_chunk_count": len(gold_chunk_set),
        "gold_contract_count": len(gold_title_set),
        "retrieved_ids": ids,
        "retrieved_titles": titles,
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "reps": len(latencies),
        },
        "chunk_metrics": compute_metrics(ids, gold_chunk_set),
        "contract_metrics": compute_metrics(titles, gold_title_set),
        "span_hit": _compute_span_hit(results, gold_spans_for_query),
        "top_highlights": [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "highlight": (r.get("highlight") or r.get("highlights") or [None])[:1],
            }
            for r in results[:5]
        ],
        "error": last_err,
    }

    if do_chat:
        try:
            out["chat"] = call_chat(client, base_url, q["q"])
        except Exception as e:
            out["chat"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def macro_avg(per_query: list[dict], metrics_key: str, k: str) -> float | None:
    vals = [
        pq[metrics_key].get(k)
        for pq in per_query
        if pq.get(metrics_key) and pq[metrics_key].get(k) is not None
    ]
    return statistics.mean(vals) if vals else None


def summarize(per_query: list[dict], gold_contracts: dict[str, dict[str, Any]]) -> dict:
    summary: dict[str, Any] = {
        "n_queries": len(per_query),
        "chunk_metrics": {k: macro_avg(per_query, "chunk_metrics", k) for k in METRIC_KEYS},
        "contract_metrics": {k: macro_avg(per_query, "contract_metrics", k) for k in METRIC_KEYS},
    }

    # Span-Hit: fraction of queries where the engine returned a chunk whose
    # char window overlaps the CUAD-annotated span (None queries are excluded).
    for k in ("hit@1", "hit@5", "hit@10"):
        vals = [
            pq["span_hit"][k]
            for pq in per_query
            if pq.get("span_hit") and pq["span_hit"].get(k) is not None
        ]
        summary.setdefault("span_hit", {})[k] = (
            sum(vals) / len(vals) if vals else None
        )
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
        "p95_max_across_queries": max(lat_p95) if lat_p95 else None,
    }
    summary["zero_result_queries"] = [pq["id"] for pq in per_query if not pq["retrieved_ids"]]

    # /chat source-title coverage if present. Uses gold_contracts (titles).
    chat_rows = [
        pq for pq in per_query if "chat" in pq and isinstance(pq.get("chat"), dict) and "answer" in pq["chat"]
    ]
    if chat_rows:
        covered = 0
        cannot = 0
        for pq in chat_rows:
            titles_in_chat = {
                s.get("title") for s in pq["chat"].get("sources") or [] if s.get("title")
            }
            gold_titles = set(
                (gold_contracts.get(pq["id"]) or {}).get("relevant_titles") or []
            )
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
    ap.add_argument("--strategy", default="semantic_search",
                    choices=["semantic_search", "hybrid_search"],
                    help="Search strategy passed to /search (default: semantic_search)")
    ap.add_argument(
        "--gold-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory containing gold.json / gold_contracts.json / gold_spans.json "
             "(default: tests/eval/). Pass the --out-dir used by build_gold.py.",
    )
    args = ap.parse_args()

    gold_path          = args.gold_dir / "gold.json"
    gold_contracts_path = args.gold_dir / "gold_contracts.json"
    gold_spans_path    = args.gold_dir / "gold_spans.json"

    if not args.queries.exists():
        log.error("queries file not found: %s", args.queries)
        return 2
    queries = json.loads(args.queries.read_text())
    log.info("Loaded %d queries from %s", len(queries), args.queries)

    gold: dict[str, list] = {}
    if gold_path.exists():
        gold = json.loads(gold_path.read_text())
        log.info("Loaded chunk-level gold for %d query ids from %s", len(gold), gold_path)
    else:
        log.warning("No gold.json at %s; chunk metrics will be null. Run build_gold.py first.", gold_path)

    gold_contracts: dict[str, dict[str, Any]] = {}
    if gold_contracts_path.exists():
        gold_contracts = json.loads(gold_contracts_path.read_text())
        log.info("Loaded contract-level gold for %d query ids from %s", len(gold_contracts), gold_contracts_path)
    else:
        log.warning(
            "No gold_contracts.json at %s; contract metrics will be null. "
            "Run build_gold.py first.", gold_contracts_path
        )

    gold_spans: dict[str, dict[str, Any]] = {}
    if gold_spans_path.exists():
        gold_spans = json.loads(gold_spans_path.read_text())
        log.info("Loaded span-level gold for %d query ids from %s", len(gold_spans), gold_spans_path)
    else:
        log.warning("No gold_spans.json at %s; span_hit metrics will be null. Run build_gold.py first.", gold_spans_path)

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
            gold_chunks = gold.get(qid, [])
            gold_titles = (gold_contracts.get(qid) or {}).get("relevant_titles") or []
            gold_spans_for_query = (gold_spans.get(qid) or {}).get("spans_by_document") or {}
            log.info(
                "[%d/%d] %s (chunks=%d docs=%d)",
                i + 1,
                len(queries),
                qid,
                len(gold_chunks),
                len(gold_titles),
            )
            row = evaluate_query(
                client,
                args.base_url,
                q,
                gold_chunks,
                gold_titles,
                gold_spans_for_query,
                do_chat=args.chat,
                strategy=args.strategy,
            )
            per_query.append(row)

    (out_dir / "search.json").write_text(json.dumps(per_query, indent=2))
    summary = summarize(per_query, gold_contracts)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Summary written to %s/summary.json", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
