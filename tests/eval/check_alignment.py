"""
check_alignment.py — Compare Qdrant-indexed titles vs CUAD gold.

Scrolls the cuad_contracts Qdrant collection, reads gold_contracts.json, and
reports:
  - Titles in Qdrant but absent from CUAD gold relevant_titles (not annotated)
  - Titles in CUAD gold but absent from Qdrant (annotated but not indexed)
  - Per-query coverage: fraction of relevant titles that are indexed
  - Overall coverage summary

Requires a live Qdrant connection (reads QDRANT_URL / CLUSTER_URL from .env).

Usage:
  python tests/eval/check_alignment.py
  python tests/eval/check_alignment.py --verbose
  python tests/eval/check_alignment.py --collection my_collection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cuad-demo-quadrant"))

from qdrant_cluster_connect import get_qdrant_client  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLD_CONTRACTS_PATH = EVAL_DIR / "gold_contracts.json"
COLLECTION = "cuad_contracts"


def scroll_qdrant_titles(collection: str) -> set[str]:
    client = get_qdrant_client()
    titles: set[str] = set()
    offset = None
    pages = 0
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=["title"],
            with_vectors=False,
        )
        for p in batch:
            t = (p.payload or {}).get("title", "").strip()
            if t:
                titles.add(t)
        pages += 1
        if offset is None:
            break
    print(f"  Scrolled {len(titles)} unique titles from '{collection}' ({pages} pages)")
    return titles


def load_gold_contracts() -> dict:
    if not GOLD_CONTRACTS_PATH.exists():
        print("  gold_contracts.json: MISSING — run build_gold.py first", file=sys.stderr)
        sys.exit(2)
    return json.loads(GOLD_CONTRACTS_PATH.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Qdrant ↔ CUAD gold title alignment.")
    ap.add_argument("--verbose", "-v", action="store_true", help="List all missing/extra titles")
    ap.add_argument("--collection", default=COLLECTION, help="Qdrant collection name")
    args = ap.parse_args()

    print("=== Qdrant scroll ===")
    qdrant_titles = scroll_qdrant_titles(args.collection)

    print("\n=== Gold contracts ===")
    gc = load_gold_contracts()
    all_gold_titles: set[str] = set()
    for entry in gc.values():
        all_gold_titles.update(entry.get("relevant_titles") or [])
    print(f"  gold_contracts.json: {len(gc)} queries, {len(all_gold_titles)} unique relevant titles")

    print("\n=== Title alignment ===")
    in_qdrant_not_gold = qdrant_titles - all_gold_titles
    in_gold_not_qdrant = all_gold_titles - qdrant_titles
    in_both = qdrant_titles & all_gold_titles

    print(f"  Indexed  & in gold:     {len(in_both)}")
    print(f"  Indexed  & NOT in gold: {len(in_qdrant_not_gold)}  (extra — ingested but CUAD has no annotations for them)")
    print(f"  In gold  & NOT indexed: {len(in_gold_not_qdrant)}  (gap — annotated but missing from Qdrant)")

    if in_qdrant_not_gold:
        if args.verbose:
            print("\n  Extra titles (in Qdrant, not in CUAD gold):")
            for t in sorted(in_qdrant_not_gold):
                print(f"    {t}")
        else:
            sample = sorted(in_qdrant_not_gold)[:5]
            print(f"\n  Extra titles sample: {sample}{'...' if len(in_qdrant_not_gold) > 5 else ''}")

    if in_gold_not_qdrant:
        if args.verbose:
            print("\n  Missing titles (in CUAD gold, not indexed):")
            for t in sorted(in_gold_not_qdrant):
                print(f"    {t}")
        else:
            sample = sorted(in_gold_not_qdrant)[:5]
            print(f"\n  Missing titles sample: {sample}{'...' if len(in_gold_not_qdrant) > 5 else ''}")

    print("\n=== Per-query coverage ===")
    coverages = []
    zero_coverage = []
    partial_coverage = []
    for qid, entry in gc.items():
        relevant = set(entry.get("relevant_titles") or [])
        if not relevant:
            continue
        covered = relevant & qdrant_titles
        cov = len(covered) / len(relevant)
        coverages.append(cov)
        if cov == 0.0:
            zero_coverage.append(qid)
        elif cov < 1.0:
            partial_coverage.append((qid, len(covered), len(relevant), cov))

    if coverages:
        import statistics
        print(f"  mean coverage:    {statistics.mean(coverages):.1%}")
        print(f"  median coverage:  {statistics.median(coverages):.1%}")
        print(f"  min coverage:     {min(coverages):.1%}")
        print(f"  queries at 100%:  {sum(1 for c in coverages if c == 1.0)}/{len(coverages)}")
        print(f"  queries at 0%:    {len(zero_coverage)}")
        print(f"  queries partial:  {len(partial_coverage)}")

    if zero_coverage:
        print(f"\n  Zero-coverage queries: {zero_coverage}")

    if partial_coverage and args.verbose:
        print("\n  Partial-coverage queries:")
        for qid, covered, total, cov in sorted(partial_coverage, key=lambda x: x[3]):
            print(f"    {qid}: {covered}/{total} ({cov:.1%})")

    print("\n=== Summary ===")
    total_gold = len(all_gold_titles)
    covered_gold = len(in_both)
    print(f"  {covered_gold}/{total_gold} CUAD-annotated titles are indexed ({100*covered_gold//total_gold if total_gold else 0}%)")
    if in_gold_not_qdrant:
        print(f"  ACTION: {len(in_gold_not_qdrant)} titles need ingestion — see INGEST_HANDOFF.md")
    else:
        print("  All annotated titles are indexed.")


if __name__ == "__main__":
    main()
