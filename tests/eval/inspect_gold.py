"""
inspect_gold.py — Stats dashboard for the eval gold artifacts.

Reads gold.json, gold_contracts.json, gold_spans.json, queries.json, and
doc_scoped_titles.json and prints a summary for each file plus a cross-file
alignment check.

Usage:
  python tests/eval/inspect_gold.py
  python tests/eval/inspect_gold.py --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def _pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n // total}%)" if total else f"{n}/0"


def _dist(values: list[int], label: str) -> str:
    if not values:
        return f"{label}: (empty)"
    return (
        f"{label}: min={min(values)} "
        f"median={int(statistics.median(values))} "
        f"max={max(values)} "
        f"mean={statistics.mean(values):.1f}"
    )


def inspect_queries(verbose: bool) -> set[str]:
    path = EVAL_DIR / "queries.json"
    if not path.exists():
        print("  queries.json: MISSING")
        return set()
    qs = json.loads(path.read_text())
    ids = {q["id"] for q in qs}
    forms: dict[str, int] = {}
    cats: set[str] = set()
    for q in qs:
        forms[q.get("form", "?")] = forms.get(q.get("form", "?"), 0) + 1
        cats.add(q.get("category", "?"))
    print(f"  queries.json:        {len(qs)} queries  |  forms: {dict(sorted(forms.items()))}  |  {len(cats)} unique categories")
    if verbose:
        for q in qs[:5]:
            print(f"    [{q['id']}] form={q.get('form')} cat={q.get('category')}")
        if len(qs) > 5:
            print(f"    ... {len(qs)-5} more")
    return ids


def inspect_gold_chunks(query_ids: set[str], verbose: bool) -> set[str]:
    path = EVAL_DIR / "gold.json"
    if not path.exists():
        print("  gold.json:           MISSING — run build_gold.py")
        return set()
    gold = json.loads(path.read_text())
    sizes = [len(v) for v in gold.values()]
    zeros = sum(1 for s in sizes if s == 0)
    extra = set(gold.keys()) - query_ids
    missing = query_ids - set(gold.keys())
    print(f"  gold.json:           {len(gold)} entries  |  {_dist(sizes, 'chunks')}  |  zero={zeros}")
    if extra:
        print(f"    WARNING: {len(extra)} qids in gold not in queries.json: {sorted(extra)[:3]}...")
    if missing:
        print(f"    WARNING: {len(missing)} queries missing from gold — rebuild needed")
    if verbose:
        for qid, pts in list(gold.items())[:3]:
            print(f"    {qid}: {len(pts)} chunks")
    return set(gold.keys())


def inspect_gold_contracts(query_ids: set[str], verbose: bool) -> set[str]:
    path = EVAL_DIR / "gold_contracts.json"
    if not path.exists():
        print("  gold_contracts.json: MISSING — run build_gold.py")
        return set()
    gc = json.loads(path.read_text())
    sizes = [v["total_relevant"] for v in gc.values()]
    zeros = sum(1 for s in sizes if s == 0)
    all_titles: set[str] = set()
    for v in gc.values():
        all_titles.update(v.get("relevant_titles") or [])
    extra = set(gc.keys()) - query_ids
    missing = query_ids - set(gc.keys())
    print(
        f"  gold_contracts.json: {len(gc)} entries  |  {_dist(sizes, 'relevant_docs')}  |  "
        f"zero={zeros}  |  {len(all_titles)} unique titles across all queries"
    )
    if extra:
        print(f"    WARNING: {len(extra)} qids in gold_contracts not in queries.json")
    if missing:
        print(f"    WARNING: {len(missing)} queries missing from gold_contracts — rebuild needed")
    if verbose:
        for qid, entry in list(gc.items())[:3]:
            print(f"    {qid}: {entry['total_relevant']} relevant titles (category={entry['category']})")
    return set(gc.keys())


def inspect_gold_spans(query_ids: set[str], verbose: bool) -> None:
    path = EVAL_DIR / "gold_spans.json"
    if not path.exists():
        print("  gold_spans.json:     MISSING — run build_gold.py")
        return
    gs = json.loads(path.read_text())
    total_spans = sum(
        sum(len(v) for v in e["spans_by_document"].values()) for e in gs.values()
    )
    spans_per_query = [
        sum(len(v) for v in e["spans_by_document"].values()) for e in gs.values()
    ]
    zeros = sum(1 for s in spans_per_query if s == 0)
    largest_group = max(
        (len(spans) for e in gs.values() for spans in e["spans_by_document"].values()),
        default=0,
    )
    docs_per_query = [len(e["spans_by_document"]) for e in gs.values()]
    print(
        f"  gold_spans.json:     {len(gs)} entries  |  {total_spans} total spans  |  "
        f"zero_span_queries={zeros}  |  largest_(qid,doc)_group={largest_group}  |  "
        f"{_dist(docs_per_query, 'docs_per_query')}"
    )
    if verbose:
        for qid, entry in list(gs.items())[:3]:
            n = sum(len(v) for v in entry["spans_by_document"].values())
            print(f"    {qid}: {n} spans across {len(entry['spans_by_document'])} docs")


def inspect_doc_scoped(verbose: bool) -> None:
    path = EVAL_DIR / "doc_scoped_titles.json"
    if not path.exists():
        print("  doc_scoped_titles.json: MISSING — run build_gold.py")
        return
    titles = json.loads(path.read_text())
    print(f"  doc_scoped_titles.json: {len(titles)} sampled titles")
    if verbose:
        for t in titles:
            print(f"    {t}")


def inspect_runs(verbose: bool) -> None:
    runs_dir = EVAL_DIR / "runs"
    if not runs_dir.exists():
        print("  runs/:               no eval runs yet — run run_eval.py")
        return
    runs = sorted(runs_dir.iterdir(), reverse=True)
    print(f"  runs/:               {len(runs)} run(s)")
    for run_dir in runs[:3]:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"    {run_dir.name}: no summary.json")
            continue
        s = json.loads(summary_path.read_text())
        cm = s.get("contract_metrics", {})
        lat = s.get("latency_s", {})
        r10 = cm.get("recall@10")
        ndcg = cm.get("ndcg@10")
        p50 = lat.get("p50_median_across_queries")
        print(
            f"    {run_dir.name}: "
            f"contract recall@10={r10:.3f}  ndcg@10={ndcg:.3f}  "
            f"p50={p50:.2f}s"
            if (r10 is not None and ndcg is not None and p50 is not None)
            else f"    {run_dir.name}: (incomplete summary)"
        )
        if verbose:
            for k in ("recall@5", "recall@20", "mrr@20"):
                v = cm.get(k)
                if v is not None:
                    print(f"      contract_{k}={v:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print eval gold artifact stats.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show sample entries")
    args = ap.parse_args()

    print("=== queries ===")
    query_ids = inspect_queries(args.verbose)

    print("\n=== gold artifacts ===")
    inspect_gold_chunks(query_ids, args.verbose)
    inspect_gold_contracts(query_ids, args.verbose)
    inspect_gold_spans(query_ids, args.verbose)
    inspect_doc_scoped(args.verbose)

    print("\n=== eval runs ===")
    inspect_runs(args.verbose)


if __name__ == "__main__":
    main()
