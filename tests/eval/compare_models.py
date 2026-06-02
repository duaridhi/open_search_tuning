#!/usr/bin/env python3
"""
compare_models.py  —  Per-document retrieval comparison across 4 embedding models.

Fires sample queries scoped to a single contract document against 4 running
servers, shows retrieved chunks vs CUAD gold annotations, and prints a ranked
comparison table.

Usage
-----
  python tests/eval/compare_models.py                      # auto-pick doc
  python tests/eval/compare_models.py --doc "DigitalCinema"
  python tests/eval/compare_models.py --list-docs          # ranked by coverage

Requires 4 model servers; start them with:
  source .env.dev && bash tests/eval/start_eval_servers.sh
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────

EVAL_DIR  = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent

# (query_id_prefix, display_label, query_text_sent_to_server)
SAMPLE_QUERIES: list[tuple[str, str, str]] = [
    ("anti_assignment",
     "Anti-Assignment",
     'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be'
     " reviewed by a lawyer. Details: Is consent or notice required of a party if the contract"
     " is assigned to a third party?"),
    ("license_grant",
     "License Grant",
     'Highlight the parts (if any) of this contract related to "License Grant" that should be'
     " reviewed by a lawyer. Details: Does the contract contain a license granted by one party"
     " to its counterparty?"),
    ("governing_law",
     "Governing Law",
     'Highlight the parts (if any) of this contract related to "Governing Law" that should be'
     " reviewed by a lawyer. Details: Which state/country's law governs the interpretation of"
     " the contract?"),
    ("audit_rights",
     "Audit Rights",
     'Highlight the parts (if any) of this contract related to "Audit Rights" that should be'
     " reviewed by a lawyer. Details: Does a party have the right to audit the books, records,"
     " or physical locations of the counterparty to ensure compliance with the contract?"),
    ("revenue_profit_sharing",
     "Revenue/Profit Sharing",
     'Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing" that'
     " should be reviewed by a lawyer. Details: Is one party required to share revenue or profit"
     " with the counterparty for any technology, goods, or services?"),
]

MODEL_SERVERS: dict[str, dict] = {
    "voyage-law-2":  {"url": "http://localhost:8006", "collection": "cuad_voyage_law2_hybrid_10",  "strategy": "hybrid_search"},
    "voy-law2-200":  {"url": "http://localhost:8007", "collection": "cuad_voyage_law2_hybrid_200", "strategy": "hybrid_search"},
    "bge-large":     {"url": "http://localhost:8013", "collection": "cuad_bgelarge_hybrid_50",      "strategy": "hybrid_search"},
    "mpnet":         {"url": "http://localhost:8012", "collection": "cuad_mpnet_hybrid_50",         "strategy": "hybrid_search"},
    "minilm":        {"url": "http://localhost:8011", "collection": "cuad_minilm_hybrid_50",        "strategy": "hybrid_search"},
}

# Additional sub-queries for multi-annotation categories.
# When --multi-query is set, each model fires all variants per category and
# results are union-merged by deduplication before metric computation.
QUERY_VARIANTS: dict[str, list[str]] = {
    "license_grant": [
        "Does either party grant a non-exclusive license to use trademarks, brand names, or marks?",
        "Are there sublicense rights, content licenses, or digital distribution licenses in this contract?",
    ],
    "audit_rights": [
        "What restrictions or limitations apply to audit rights? How long after the audit period ends?",
        "Can one party inspect financial records, physical locations, or facilities of the other party?",
    ],
    "anti_assignment": [
        "Is prior written consent required before assigning or transferring this agreement to a third party?",
    ],
}

TOP_K        = 50   # retrieve 50 so Ann@10/20/50 can be computed
MIN_SPAN_LEN = 20
ANN_KS       = (5, 10, 20, 50)

# ── Text helpers ───────────────────────────────────────────────────────────────

_CURLY = str.maketrans({0x201C: '"', 0x201D: '"', 0x2018: "'", 0x2019: "'"})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_CURLY)).strip().lower()


def _shorten(text: str, width: int) -> str:
    return textwrap.shorten(text.replace("\n", " "), width=width, placeholder="…")


def _excerpt_centred(text: str, gold_spans: list[dict], width: int = 64) -> str:
    """Return excerpt centred on first matching gold span, else start of text."""
    norm_text = _norm(text)
    for span in gold_spans:
        ans = span.get("answer_text", "")
        if len(ans) >= MIN_SPAN_LEN:
            idx = norm_text.find(_norm(ans[:50]))
            if idx >= 0:
                raw_start = max(0, idx - 30)
                return _shorten(text[raw_start: raw_start + width], width)
    return _shorten(text, width)


# ── Metric helpers ─────────────────────────────────────────────────────────────

_SPAN_WINDOW = 100  # chars: for spans > this, check sliding windows (handles small chunks)


def _span_in_chunk(span_text: str, chunk_text: str) -> bool:
    """True if span is fully contained in chunk OR any 100-char window of span is in chunk.

    The sliding-window fallback handles collections with small chunk sizes where a
    long gold span straddles two adjacent chunks — each chunk still contains a
    substantial portion of the span.
    """
    norm_span  = _norm(span_text)
    norm_chunk = _norm(chunk_text)
    if norm_span in norm_chunk:
        return True
    if len(norm_span) > _SPAN_WINDOW:
        for i in range(0, len(norm_span) - _SPAN_WINDOW + 1, 40):
            if norm_span[i : i + _SPAN_WINDOW] in norm_chunk:
                return True
    return False


def find_hit_rank(retrieved: list[dict], gold_spans: list[dict]) -> int | None:
    """Rank of first retrieved chunk that contains any gold span; None if miss."""
    for rank, chunk in enumerate(retrieved, 1):
        chunk_text = chunk.get("text") or ""
        for span in gold_spans:
            ans = span.get("answer_text", "")
            if len(ans) >= MIN_SPAN_LEN and _span_in_chunk(ans, chunk_text):
                return rank
    return None


def annotation_recall_at_k(
    retrieved: list[dict], gold_spans: list[dict], k: int
) -> tuple[int, int]:
    """
    (found, total): how many distinct gold spans appear in at least one top-k chunk.
    Returns (0, 0) when there are no valid gold spans (query has no annotations for doc).
    """
    valid = [s for s in gold_spans if len(s.get("answer_text", "")) >= MIN_SPAN_LEN]
    if not valid:
        return (0, 0)
    found = sum(
        1 for span in valid
        if any(_span_in_chunk(span["answer_text"], c.get("text") or "") for c in retrieved[:k])
    )
    return (found, len(valid))


def spans_in_chunk(chunk: dict, gold_spans: list[dict]) -> list[int]:
    """Return 1-indexed positions of gold spans found in this chunk."""
    chunk_text = chunk.get("text") or ""
    return [
        i for i, span in enumerate(gold_spans, 1)
        if len(span.get("answer_text", "")) >= MIN_SPAN_LEN
        and _span_in_chunk(span["answer_text"], chunk_text)
    ]


# ── Gold helpers ───────────────────────────────────────────────────────────────

def load_gold_spans() -> dict:
    candidates = [
        EVAL_DIR / "gold" / "cuad_voyage_law2_hybrid_10" / "gold_spans.json",
        EVAL_DIR / "gold_spans.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return {}


def resolve_doc(partial: str, gold_spans: dict) -> str | None:
    all_docs: set[str] = set()
    for val in gold_spans.values():
        all_docs.update((val.get("spans_by_document") or {}).keys())
    key = partial.lower()
    matches = [d for d in all_docs if key in d.lower()]
    if not matches:
        return None
    for m in matches:
        if m.lower().startswith(key):
            return m
    return matches[0]


def select_docs(gold_spans: dict, queries: list[tuple], n: int) -> list[str]:
    """Return up to n docs ranked by total valid gold span count for the selected queries."""
    q_prefixes = {q[0] for q in queries}
    score: dict[str, int] = defaultdict(int)
    for qid, val in gold_spans.items():
        prefix = qid.rsplit("__", 1)[0]
        if prefix not in q_prefixes:
            continue
        for doc, spans in (val.get("spans_by_document") or {}).items():
            score[doc] += sum(1 for s in spans if len(s.get("answer_text", "")) >= MIN_SPAN_LEN)
    return sorted(score, key=score.__getitem__, reverse=True)[:n]


def pick_best_doc(gold_spans: dict, queries: list[tuple]) -> str:
    q_prefixes = {q[0] for q in queries}
    coverage: dict[str, int] = defaultdict(int)
    for qid, val in gold_spans.items():
        prefix = qid.rsplit("__", 1)[0]
        if prefix in q_prefixes:
            for doc in (val.get("spans_by_document") or {}):
                coverage[doc] += 1
    if not coverage:
        for val in gold_spans.values():
            for doc in (val.get("spans_by_document") or {}):
                coverage[doc] += 1
    return max(coverage, key=coverage.__getitem__)


def get_gold_spans_for_doc(gold_spans: dict, doc: str, qid_prefix: str) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for full_qid, val in gold_spans.items():
        if full_qid.rsplit("__", 1)[0] != qid_prefix:
            continue
        for span in (val.get("spans_by_document") or {}).get(doc, []):
            key = _norm(span.get("answer_text", ""))
            if key and key not in seen:
                seen.add(key)
                result.append(span)
    return result


# ── HTTP ───────────────────────────────────────────────────────────────────────

def search(client: httpx.Client, base_url: str, q: str, doc: str,
           strategy: str, top_k: int) -> list[dict]:
    try:
        resp = client.get(
            f"{base_url}/search",
            params={"q": q, "top_k": top_k, "document_name": doc, "strategy": strategy},
            timeout=120.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("results") or payload.get("hits") or []
    except Exception as exc:
        return [{"_error": str(exc)[:100]}]


def search_multi(
    client: httpx.Client, base_url: str, main_query: str, variants: list[str],
    doc: str, strategy: str, top_k: int,
) -> list[dict]:
    """Fire main query + variants, deduplicate by normalised text prefix, return union.

    If the main query fails, propagates the error. If a variant fails, skips it
    so one bad sub-query doesn't void the entire merged result.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for i, q in enumerate([main_query] + variants):
        results = search(client, base_url, q, doc, strategy, top_k)
        if results and results[0].get("_error"):
            if i == 0:
                return results  # main query failed — propagate
            continue  # variant failed — skip, keep partial merged result
        for r in results:
            key = _norm(r.get("text", ""))[:120]
            if key not in seen:
                seen.add(key)
                merged.append(r)
    return merged


def ping(client: httpx.Client, base_url: str) -> bool:
    try:
        client.get(f"{base_url}/health", timeout=5.0).raise_for_status()
        return True
    except Exception:
        return False


# ── Display ────────────────────────────────────────────────────────────────────

# Column widths for the per-query table
_WM = 16   # model name
_WR = 6    # rank
_WA = 5    # Ann@5  (shows "f/t" e.g. "3/3")
_WT = 64   # chunk excerpt

def _table_top() -> str:
    return f"  ┌{'─'*(_WM+2)}┬{'─'*(_WR+2)}┬{'─'*(_WA+2)}┬{'─'*(_WT+2)}┐"

def _table_hdr() -> str:
    return (f"  │ {'Model':<{_WM}} │ {'Rank':>{_WR}} │ {'Ann@5':^{_WA}} │ "
            f"{'Retrieved chunk (best hit excerpt)':<{_WT}} │")

def _table_div() -> str:
    return f"  ├{'─'*(_WM+2)}┼{'─'*(_WR+2)}┼{'─'*(_WA+2)}┼{'─'*(_WT+2)}┤"

def _table_bot() -> str:
    return f"  └{'─'*(_WM+2)}┴{'─'*(_WR+2)}┴{'─'*(_WA+2)}┴{'─'*(_WT+2)}┘"

def _table_row(model: str, rank_str: str, ann5: str, excerpt: str) -> str:
    return (f"  │ {model:<{_WM}} │ {rank_str:>{_WR}} │ {ann5:^{_WA}} │ {excerpt:<{_WT}} │")


def compute_query_metrics(
    model_results: dict[str, list[dict]],
    gold_spans: list[dict],
) -> tuple[dict[str, int | None], dict[str, dict[int, tuple[int, int]]]]:
    """Compute ranks and annotation recalls without printing anything."""
    ranks: dict[str, int | None] = {}
    ann_recalls: dict[str, dict[int, tuple[int, int]]] = {}
    for model, results in model_results.items():
        if results and results[0].get("_error"):
            ranks[model] = None
            ann_recalls[model] = {k: (0, 0) for k in ANN_KS}
        else:
            ranks[model] = find_hit_rank(results, gold_spans)
            ann_recalls[model] = {k: annotation_recall_at_k(results, gold_spans, k) for k in ANN_KS}
    return ranks, ann_recalls


def print_query_section(
    qlabel: str,
    doc: str,
    gold_spans: list[dict],
    model_results: dict[str, list[dict]],
    ranks: dict[str, int | None],
    ann_recalls: dict[str, dict[int, tuple[int, int]]],
) -> None:
    """Display per-query table using pre-computed ranks and ann_recalls."""
    total_w = _WM + _WR + _WA + _WT + 16
    thick = "═" * total_w
    print()
    print(f"  {thick}")
    print(f"  QUERY : {qlabel}")
    print(f"  DOC   : {doc[:total_w - 10]}")
    print(f"  {thick}")

    n_spans = len([s for s in gold_spans if len(s.get("answer_text", "")) >= MIN_SPAN_LEN])
    if gold_spans:
        print(f"\n  GOLD ANNOTATIONS ({n_spans} total):")
        for i, span in enumerate(gold_spans, 1):
            ans = span.get("answer_text", "")
            print(f"    [{i}] \"{_shorten(ans, 160)}\"")
    else:
        print("\n  GOLD ANNOTATIONS: none for this document / query combination")

    print()
    print(_table_top())
    print(_table_hdr())
    print(_table_div())

    for model, results in model_results.items():
        if results and results[0].get("_error"):
            print(_table_row(model, "ERR", "—", results[0]["_error"][:_WT]))
            continue

        rank  = ranks.get(model)
        ann_k = ann_recalls.get(model, {})

        if not results:
            rank_str = "  -"
            ann5_str = "—"
            excerpt  = "(no results)"
        else:
            rank_str = f"{rank} ✓" if rank is not None else "  -"
            f5, t5   = ann_k.get(5, (0, 0))
            ann5_str = f"{f5}/{t5}" if t5 > 0 else "—"
            best_idx = (rank - 1) if rank is not None else 0
            best_idx = min(best_idx, len(results) - 1)
            excerpt  = _excerpt_centred(results[best_idx].get("text") or "", gold_spans)

        print(_table_row(model, rank_str, ann5_str, excerpt))

    print(_table_bot())


def print_all_chunks(
    qlabel: str,
    doc: str,
    gold_spans: list[dict],
    model_results: dict[str, list[dict]],
    top_k: int,
) -> None:
    """Print full chunk text for each model with per-chunk annotation hit markers."""
    total_w = 90
    n_spans = len([s for s in gold_spans if len(s.get("answer_text", "")) >= MIN_SPAN_LEN])
    print()
    print(f"  {'─'*total_w}")
    print(f"  DETAILED CHUNKS  —  {qlabel}  ({n_spans} gold annotations)  |  {doc[:45]}")
    print(f"  {'─'*total_w}")

    if gold_spans:
        print(f"\n  GOLD ({n_spans} annotations):")
        for i, span in enumerate(gold_spans, 1):
            ans = span.get("answer_text", "")
            wrapped = textwrap.fill(ans, width=86, initial_indent=f"    [{i}] ",
                                    subsequent_indent="         ")
            print(wrapped)

    for model, results in model_results.items():
        if results and results[0].get("_error"):
            print(f"\n  ── {model} ──\n    ERROR: {results[0]['_error']}")
            continue

        ar = {k: annotation_recall_at_k(results, gold_spans, k) for k in ANN_KS}
        ar_str = "  ".join(
            f"Ann@{k}: {f}/{t}" if t > 0 else f"Ann@{k}: —"
            for k, (f, t) in ar.items()
        )
        print(f"\n  ── {model}  [{ar_str}] ──")

        for i, chunk in enumerate(results[:top_k], 1):
            text  = chunk.get("text") or ""
            hits  = spans_in_chunk(chunk, gold_spans)
            if hits:
                marker = f" ✓[{','.join(map(str,hits))}]"
            else:
                marker = "    "
            wrapped = textwrap.fill(
                text, width=86,
                initial_indent=f"    [{i:2d}]{marker} ",
                subsequent_indent="            ",
            )
            print(wrapped)


def print_per_query_ann_tables(
    all_ranks: dict[str, dict[str, int | None]],
    all_ann_recalls: dict[str, dict[str, dict[int, tuple[int, int]]]],
    queries: list[tuple],
    doc: str,
) -> None:
    """One compact table per query: Rank + Ann@5 / Ann@10 / Ann@20 per model."""
    model_names = list(MODEL_SERVERS.keys())
    W_R, W_A = 6, 9
    total_w = _WM + W_R + W_A * len(ANN_KS) + 8 + 2 * len(ANN_KS)

    print()
    print("  " + "═" * total_w)
    print("  PER-QUERY ANNOTATION RECALL  —  Rank + fraction of gold spans found at each k")
    print(f"  Document: {doc[:total_w - 12]}")

    for qid_prefix, qlabel, _ in queries:
        n_spans = 0
        for model in model_names:
            _, t = (all_ann_recalls.get(model) or {}).get(qid_prefix, {}).get(5, (0, 0))
            if t > 0:
                n_spans = t
                break

        print()
        print(f"  {'─'*total_w}")
        print(f"  {qlabel}  ({n_spans} gold annotation{'s' if n_spans != 1 else ''})")
        print(f"  {'─'*total_w}")
        ann_hdr = "  ".join(f"{'Ann@'+str(k):^{W_A}}" for k in ANN_KS)
        print(f"  {'Model':<{_WM}}  {'Rank':>{W_R}}  {ann_hdr}")
        print(f"  {'─'*total_w}")

        for model in model_names:
            rank = (all_ranks.get(model) or {}).get(qid_prefix)
            rank_str = f"{rank} ✓" if rank is not None else "-"
            ann_k = (all_ann_recalls.get(model) or {}).get(qid_prefix) or {}
            cells = []
            for k in ANN_KS:
                f, t = ann_k.get(k, (0, 0))
                cells.append(f"{f}/{t}" if t > 0 else "—")
            ann_vals = "  ".join(f"{c:^{W_A}}" for c in cells)
            print(f"  {model:<{_WM}}  {rank_str:>{W_R}}  {ann_vals}")

    print(f"  {'─'*total_w}")
    print()


def print_summary(
    all_ranks: dict[str, dict[str, int | None]],
    all_ann_recalls: dict[str, dict[str, dict[int, tuple[int, int]]]],
    queries: list[tuple],
    doc: str,
    top_k: int,
) -> None:
    model_names = list(MODEL_SERVERS.keys())
    n_q = len(queries)
    q_cols = [q[1][:13] for q in queries]
    W_Q, W_STAT = 13, 7
    total_w = 2 + _WM + (W_Q + 2) * n_q + (W_STAT + 2) * 3

    # ── Rank / hit@k table ─────────────────────────────────────────────────────
    print()
    print("  " + "═" * total_w)
    print(f"  RETRIEVAL SUMMARY  —  rank of first gold span hit  (- = miss in top-{top_k})")
    print(f"  Document: {doc[:total_w - 12]}")
    print("  " + "─" * total_w)

    hdr = f"  {'Model':<{_WM}}"
    for ql in q_cols:
        hdr += f"  {ql:^{W_Q}}"
    hdr += f"  {'Hit@1':^{W_STAT}}  {'Hit@5':^{W_STAT}}  {'MRR':^{W_STAT}}"
    print(hdr)
    print("  " + "─" * total_w)

    for model in model_names:
        q_ranks = all_ranks.get(model, {})
        row = f"  {model:<{_WM}}"
        ranks: list[int | None] = []
        for q in queries:
            r = q_ranks.get(q[0])
            ranks.append(r)
            cell = f"{r} ✓" if (r is not None and r <= top_k) else "-"
            row += f"  {cell:^{W_Q}}"
        hit1 = sum(1 for r in ranks if r == 1) / n_q
        hit5 = sum(1 for r in ranks if r is not None and r <= 5) / n_q
        mrr  = sum(1.0 / r for r in ranks if r is not None) / n_q
        row += f"  {hit1:.2f}   {hit5:.2f}   {mrr:.3f}"
        print(row)

    print("  " + "─" * total_w)

    # ── Annotation recall table ────────────────────────────────────────────────
    W_ANN = 8
    ann_total_w = 2 + _WM + (W_Q + 2) * n_q + (W_ANN + 2) * len(ANN_KS)
    print()
    print("  " + "═" * ann_total_w)
    print("  ANNOTATION RECALL  —  fraction of gold spans found across top-k chunks")
    print(f"  Document: {doc[:ann_total_w - 12]}")
    print("  " + "─" * ann_total_w)

    hdr2 = f"  {'Model':<{_WM}}"
    for ql in q_cols:
        hdr2 += f"  {ql:^{W_Q}}"
    for k in ANN_KS:
        hdr2 += f"  {'Ann@'+str(k):^{W_ANN}}"
    print(hdr2)
    print("  " + "─" * ann_total_w)

    for model in model_names:
        row = f"  {model:<{_WM}}"
        # Per-query Ann@5 cell (shows "f/t" fractions)
        for q in queries:
            ann_k = (all_ann_recalls.get(model) or {}).get(q[0]) or {}
            f5, t5 = ann_k.get(5, (0, 0))
            cell = f"{f5}/{t5}" if t5 > 0 else "—"
            row += f"  {cell:^{W_Q}}"
        # Avg Ann@5, @10, @20 across queries (skip queries with no gold spans)
        for k in ANN_KS:
            vals = []
            for q in queries:
                ann_k = (all_ann_recalls.get(model) or {}).get(q[0]) or {}
                f, t = ann_k.get(k, (0, 0))
                if t > 0:
                    vals.append(f / t)
            avg = sum(vals) / len(vals) if vals else None
            cell = f"{avg:.2f}" if avg is not None else "—"
            row += f"  {cell:^{W_ANN}}"
        print(row)

    print("  " + "─" * ann_total_w)
    print()


def print_aggregate_summary(
    agg_ranks: dict[str, list[int | None]],
    agg_ann:   dict[str, dict[int, list[float]]],
    active_models: list[str],
    n_docs: int,
    top_k: int,
) -> None:
    """Aggregate metrics across all (doc, query) pairs."""
    W_STAT, W_ANN = 8, 8
    total_w = 2 + _WM + (W_STAT + 2) * 3 + (W_ANN + 2) * len(ANN_KS)

    print()
    print("  " + "═" * total_w)
    print(f"  AGGREGATE RETRIEVAL — {n_docs} documents × {len(SAMPLE_QUERIES)} queries  (top-{top_k})")
    print("  " + "─" * total_w)
    hdr = f"  {'Model':<{_WM}}  {'Hit@1':^{W_STAT}}  {'Hit@5':^{W_STAT}}  {'MRR':^{W_STAT}}"
    for k in ANN_KS:
        hdr += f"  {'Ann@'+str(k):^{W_ANN}}"
    print(hdr)
    print("  " + "─" * total_w)

    for model in active_models:
        ranks = agg_ranks[model]
        n = len(ranks)
        if n == 0:
            print(f"  {model:<{_WM}}  (no data)")
            continue
        hit1 = sum(1 for r in ranks if r == 1) / n
        hit5 = sum(1 for r in ranks if r is not None and r <= 5) / n
        mrr  = sum(1.0 / r for r in ranks if r is not None) / n
        row  = f"  {model:<{_WM}}  {hit1:.3f}    {hit5:.3f}    {mrr:.3f} "
        for k in ANN_KS:
            vals = agg_ann[model][k]
            avg  = sum(vals) / len(vals) if vals else None
            row += f"  {avg:.3f}  " if avg is not None else f"  {'—':^{W_ANN}}"
        print(row)

    print("  " + "─" * total_w)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", default=None,
                    help="Target document (partial name, case-insensitive; default: auto)")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"Chunks to retrieve per query (default {TOP_K})")
    ap.add_argument("--list-docs", action="store_true",
                    help="List documents sorted by gold coverage and exit")
    ap.add_argument("--verbose", action="store_true",
                    help="Also print full chunk text per model per query")
    ap.add_argument("--multi-query", action="store_true",
                    help="Fire extra sub-queries for multi-annotation categories and union results")
    ap.add_argument("--num-docs", type=int, default=0,
                    help="Evaluate across N best-covered docs instead of one; 0 = single-doc mode")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model names to include (default: all up servers). "
                         "E.g. --models bge-large,mpnet,minilm")
    args = ap.parse_args()

    gold_spans = load_gold_spans()
    if not gold_spans:
        print("ERROR: gold_spans.json not found — run build_gold.py first.", file=sys.stderr)
        return 2

    if args.list_docs:
        coverage: dict[str, set[str]] = defaultdict(set)
        for qid, val in gold_spans.items():
            for doc in (val.get("spans_by_document") or {}):
                coverage[doc].add(qid.rsplit("__", 1)[0])
        print(f"\n  {'Queries':>8}  Document")
        print(f"  {'─'*8}  {'─'*80}")
        for doc, qids in sorted(coverage.items(), key=lambda x: -len(x[1]))[:20]:
            print(f"  {len(qids):>8}  {doc[:90]}")
        print()
        return 0

    # ── Apply --models filter ──────────────────────────────────────────────────
    active_servers: dict[str, dict] = dict(MODEL_SERVERS)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        unknown = wanted - set(MODEL_SERVERS)
        if unknown:
            print(f"ERROR: unknown model(s): {', '.join(sorted(unknown))}. "
                  f"Valid: {', '.join(MODEL_SERVERS)}", file=sys.stderr)
            return 2
        active_servers = {m: v for m, v in MODEL_SERVERS.items() if m in wanted}

    # ── Server health check ────────────────────────────────────────────────────
    live: dict[str, bool] = {}
    with httpx.Client() as client:
        for model, cfg in active_servers.items():
            ok = ping(client, cfg["url"])
            live[model] = ok
            status = "UP  " if ok else "DOWN"
            print(f"  [{status}]  {model:<16}  {cfg['url']}  ({cfg['collection']})")

    if not any(live.values()):
        print("\nERROR: no servers reachable. Run: source .env.dev && bash tests/eval/start_eval_servers.sh",
              file=sys.stderr)
        return 3
    downed = [m for m, ok in live.items() if not ok]
    if downed:
        print(f"\n  WARNING: {', '.join(downed)} unreachable — will show ERR rows")

    active_models = [m for m in active_servers if live.get(m)]

    # ── Multi-doc mode ─────────────────────────────────────────────────────────
    if args.num_docs > 0:
        docs = select_docs(gold_spans, SAMPLE_QUERIES, args.num_docs)
        n_docs = len(docs)
        print(f"\n  Mode       : multi-doc  ({n_docs} documents)")
        print(f"  Queries    : {', '.join(q[1] for q in SAMPLE_QUERIES)}")
        print(f"  Top-K      : {args.top_k}")
        print(f"  Multi-query: {'ON' if args.multi_query else 'OFF'}")
        est_voyageai = sum(
            1 for m in active_models
            if active_servers[m].get("collection", "").startswith("cuad_voyage")
        )
        if est_voyageai:
            print(f"  NOTE       : {est_voyageai} VoyageAI server(s) — est. "
                  f"~{n_docs * len(SAMPLE_QUERIES) * est_voyageai // 3} min at 3 req/min")
        print()

        agg_ranks: dict[str, list[int | None]]        = {m: [] for m in active_models}
        agg_ann:   dict[str, dict[int, list[float]]]  = {m: {k: [] for k in ANN_KS}
                                                         for m in active_models}

        with httpx.Client() as client:
            for doc_idx, doc in enumerate(docs, 1):
                print(f"  [{doc_idx:2d}/{n_docs}] {doc[:75]}", flush=True)
                for qid_prefix, qlabel, qtext in SAMPLE_QUERIES:
                    gold_for_doc = get_gold_spans_for_doc(gold_spans, doc, qid_prefix)
                    model_results: dict[str, list[dict]] = {}
                    for model, cfg in active_servers.items():
                        if not live[model]:
                            model_results[model] = [{"_error": "server not reachable"}]
                        elif args.multi_query and (variants := QUERY_VARIANTS.get(qid_prefix)):
                            model_results[model] = search_multi(
                                client, cfg["url"], qtext, variants, doc, cfg["strategy"], args.top_k
                            )
                        else:
                            model_results[model] = search(
                                client, cfg["url"], qtext, doc, cfg["strategy"], args.top_k
                            )
                    ranks, ann_recalls = compute_query_metrics(model_results, gold_for_doc)
                    for model in active_models:
                        agg_ranks[model].append(ranks.get(model))
                        for k in ANN_KS:
                            f, t = (ann_recalls.get(model) or {}).get(k, (0, 0))
                            if t > 0:
                                agg_ann[model][k].append(f / t)

        print_aggregate_summary(agg_ranks, agg_ann, active_models, n_docs, args.top_k)
        return 0

    # ── Single-doc mode (default) ──────────────────────────────────────────────
    if args.doc:
        doc = resolve_doc(args.doc, gold_spans)
        if not doc:
            print(f"ERROR: no document matching '{args.doc}'. Use --list-docs.", file=sys.stderr)
            return 2
    else:
        doc = pick_best_doc(gold_spans, SAMPLE_QUERIES)

    print(f"\n  Document   : {doc}")
    print(f"  Queries    : {', '.join(q[1] for q in SAMPLE_QUERIES)}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Strategy   : hybrid_search")
    print(f"  Multi-query: {'ON (extra sub-queries for multi-annotation categories)' if args.multi_query else 'OFF'}\n")

    all_ranks:       dict[str, dict[str, int | None]]                 = {m: {} for m in active_servers}
    all_ann_recalls: dict[str, dict[str, dict[int, tuple[int, int]]]] = {m: {} for m in active_servers}

    # Collect all query results and compute metrics before printing anything
    collected: list[dict] = []
    with httpx.Client() as client:
        for qid_prefix, qlabel, qtext in SAMPLE_QUERIES:
            gold_for_doc = get_gold_spans_for_doc(gold_spans, doc, qid_prefix)
            model_results: dict[str, list[dict]] = {}
            for model, cfg in active_servers.items():
                if not live[model]:
                    model_results[model] = [{"_error": "server not reachable"}]
                elif args.multi_query and (variants := QUERY_VARIANTS.get(qid_prefix)):
                    model_results[model] = search_multi(
                        client, cfg["url"], qtext, variants, doc, cfg["strategy"], args.top_k
                    )
                else:
                    model_results[model] = search(
                        client, cfg["url"], qtext, doc, cfg["strategy"], args.top_k
                    )
            ranks, ann_recalls = compute_query_metrics(model_results, gold_for_doc)
            for model in active_servers:
                all_ranks[model][qid_prefix]       = ranks.get(model)
                all_ann_recalls[model][qid_prefix] = ann_recalls.get(model, {k: (0, 0) for k in ANN_KS})
            collected.append(dict(
                qid_prefix=qid_prefix, qlabel=qlabel,
                gold_for_doc=gold_for_doc, model_results=model_results,
                ranks=ranks, ann_recalls=ann_recalls,
            ))

    # ── Print summary tables first ─────────────────────────────────────────────
    print_summary(all_ranks, all_ann_recalls, SAMPLE_QUERIES, doc, args.top_k)
    print_per_query_ann_tables(all_ranks, all_ann_recalls, SAMPLE_QUERIES, doc)

    # ── Then per-query detailed sections ──────────────────────────────────────
    for qd in collected:
        print_query_section(
            qd["qlabel"], doc, qd["gold_for_doc"], qd["model_results"],
            qd["ranks"], qd["ann_recalls"],
        )
        if args.verbose:
            print_all_chunks(qd["qlabel"], doc, qd["gold_for_doc"],
                             qd["model_results"], args.top_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
