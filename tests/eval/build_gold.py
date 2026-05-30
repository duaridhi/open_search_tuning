"""
build_gold.py — Build CUAD ground-truth artifacts for retrieval evaluation.

Emits three projections of the same underlying spans:
  1. gold.json           — chunk-level. {qid: [point_id, ...]}.
  2. gold_contracts.json — contract-level.
     {qid: {category, form, relevant_titles, total_relevant}}.
     Title-only join; primary signal for cross-contract retrieval.
  3. gold_spans.json     — raw spans per (qid, title) for future highlight eval.
     {qid: {category, form, spans_by_document: {title: [{answer_text,
            char_start, char_end}, ...]}}}.

Contract/spans projections restrict to Qdrant-ingested titles.

Usage:
  python tests/eval/build_gold.py                # rebuild everything
  python tests/eval/build_gold.py --skip-chunks  # only refresh
                                                 # gold_contracts.json +
                                                 # gold_spans.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# Make the cuad-demo-quadrant package importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cuad-demo-quadrant"))

from huggingface_hub import hf_hub_download  # noqa: E402

from qdrant_cluster_connect import get_qdrant_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_gold")

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / "cuad_gold" / "_hf_cache"
QUERIES_PATH = EVAL_DIR / "queries.json"
COLLECTION = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
DOC_SAMPLE_SIZE = 5
RNG_SEED = 42
SPAN_TEXT_TRUNC = 300

# Text-based span matching — neutralises CUAD-TXT vs PDF-extraction drift.
# Three sources of mismatch: (1) extra blank lines in TXT around section headers
# and page numbers accumulate position drift; (2) PDF preserves intra-paragraph
# line breaks that TXT joins with spaces; (3) TXT uses Unicode smart quotes while
# PDF extraction returns ASCII.  Normalising before substring search fixes all three.
_CURLY_TABLE = str.maketrans({0x201C: chr(34), 0x201D: chr(34), 0x2018: chr(39), 0x2019: chr(39)})
MIN_SPAN_TEXT_LEN = 20  # skip very short answer_texts to avoid false-positive matches


def _norm(s: str) -> str:
    """Collapse whitespace and fold curly quotes → ASCII for span matching."""
    return re.sub(r'\s+', ' ', s.translate(_CURLY_TABLE)).strip()

CAT_RE = re.compile(r'related to "([^"]+)"')


def fetch_cuad_json() -> Path:
    """Download CUAD_v1.json from HF. No fallback — raise on failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Fetching CUAD_v1.json from HF repo theatticusproject/cuad ...")
    path = hf_hub_download(
        repo_id="theatticusproject/cuad",
        filename="CUAD_v1/CUAD_v1.json",
        repo_type="dataset",
        cache_dir=str(CACHE_DIR),
    )
    log.info("CUAD_v1.json at %s", path)
    return Path(path)


def parse_spans(cuad_json_path: Path) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, int],
]:
    """Return spans and source text lengths.

    Returns:
        spans: {(title, category): [{answer_text, char_start, char_end}, ...]}
            Empty-text answers (CUAD negative rows) are dropped.
        source_lengths: {title: len(context)} — character count of the full
            contract text as annotated by CUAD; used to detect ingest truncation.
    """
    with cuad_json_path.open() as f:
        data = json.load(f)
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    source_lengths: dict[str, int] = {}
    for doc in data["data"]:
        title = doc["title"].strip()
        for para in doc["paragraphs"]:
            source_lengths[title] = max(source_lengths.get(title, 0), len(para["context"]))
            for qa in para["qas"]:
                m = CAT_RE.search(qa["question"])
                if not m:
                    continue
                cat = m.group(1)
                for ans in qa.get("answers", []):
                    text = ans.get("text", "")
                    start = ans.get("answer_start")
                    if not text or start is None:
                        continue
                    start = int(start)
                    end = start + len(text)
                    out.setdefault((title, cat), []).append(
                        {"answer_text": text, "char_start": start, "char_end": end}
                    )
    log.info("Parsed %d (title, category) span groups", len(out))
    return out, source_lengths


def scroll_points(collection_name: str = COLLECTION) -> list[dict[str, Any]]:
    """Return every point's id + title/char_start/char_end/text."""
    client = get_qdrant_client()
    points: list[dict[str, Any]] = []
    offset = None
    page = 0
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=["title", "char_start", "char_end", "text", "doc_id"],
            with_vectors=False,
        )
        for p in batch:
            pl = p.payload or {}
            title = (pl.get("title") or "").strip()
            cs = pl.get("char_start")
            ce = pl.get("char_end")
            if not title or cs is None or ce is None:
                continue
            points.append({
                "id": pl.get("doc_id") or str(p.id),
                "title": title,
                "char_start": int(cs),
                "char_end": int(ce),
                "text": pl.get("text") or "",
            })
        page += 1
        if offset is None:
            break
    log.info("Scrolled %d points across %d pages", len(points), page)
    return points


def build_gold_chunks(queries, spans, points):
    """Chunk-level: point_ids that contain a gold answer_text as a substring.

    Uses normalised text matching instead of char-offset overlap to avoid drift
    between CUAD's TXT-derived char positions and PDF-extraction char positions.
    See tests/eval/cuad_txt_vs_pdf_extract_analysis/ for the full drift analysis.
    """
    by_title: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_title.setdefault(p["title"], []).append(p)
    gold: dict[str, list] = {}
    for q in queries:
        cat = q["category"]
        seen: set = set()
        deduped: list = []
        for (title, c), span_list in spans.items():
            if c != cat:
                continue
            for p in by_title.get(title, []):
                chunk_norm = _norm(p.get("text", ""))
                for s in span_list:
                    if len(s["answer_text"]) < MIN_SPAN_TEXT_LEN:
                        continue
                    if _norm(s["answer_text"]) in chunk_norm:
                        if p["id"] not in seen:
                            seen.add(p["id"])
                            deduped.append(p["id"])
                        break
        gold[q["id"]] = deduped
    return gold


def build_gold_contracts(queries, spans, ingested_titles):
    """Contract-level: relevant_titles per query, restricted to ingested."""
    cat_to_titles: dict[str, set[str]] = {}
    for (title, cat), _ in spans.items():
        if title in ingested_titles:
            cat_to_titles.setdefault(cat, set()).add(title)
    out: dict[str, dict[str, Any]] = {}
    for q in queries:
        titles = sorted(cat_to_titles.get(q["category"], set()))
        out[q["id"]] = {
            "category": q["category"],
            "form": q.get("form"),
            "relevant_titles": titles,
            "total_relevant": len(titles),
        }
    return out


def build_gold_spans(queries, spans, ingested_titles):
    """Raw spans per (qid, title). Truncates answer_text to SPAN_TEXT_TRUNC."""
    cat_to_doc_spans: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (title, cat), span_list in spans.items():
        if title not in ingested_titles:
            continue
        per_doc = cat_to_doc_spans.setdefault(cat, {})
        bucket = per_doc.setdefault(title, [])
        for s in span_list:
            txt = s["answer_text"]
            if len(txt) > SPAN_TEXT_TRUNC:
                txt = txt[:SPAN_TEXT_TRUNC]
            bucket.append({
                "answer_text": txt,
                "char_start": s["char_start"],
                "char_end": s["char_end"],
            })
    out: dict[str, dict[str, Any]] = {}
    for q in queries:
        out[q["id"]] = {
            "category": q["category"],
            "form": q.get("form"),
            "spans_by_document": cat_to_doc_spans.get(q["category"], {}),
        }
    return out


def check_truncation(
    points: list[dict[str, Any]],
    source_lengths: dict[str, int],
    threshold: int = 200,
) -> list[dict[str, Any]]:
    """Compare max ingested char_end per title against CUAD source text length.

    Returns a list of dicts for titles where the gap exceeds *threshold*, sorted
    by gap descending.  A gap means trailing content was not ingested — clauses
    annotated beyond max_char_end will never be retrieved.
    """
    by_title: dict[str, int] = {}
    for p in points:
        t = p["title"]
        by_title[t] = max(by_title.get(t, 0), p["char_end"])

    truncated = []
    for title, src_len in source_lengths.items():
        max_ce = by_title.get(title)
        if max_ce is None:
            continue  # title not in Qdrant at all — separate issue
        gap = src_len - max_ce
        if gap > threshold:
            truncated.append({
                "title": title,
                "source_length": src_len,
                "qdrant_max_char_end": max_ce,
                "gap": gap,
            })

    truncated.sort(key=lambda x: x["gap"], reverse=True)
    if truncated:
        log.warning(
            "TRUNCATION: %d titles have gap > %d chars between CUAD source length "
            "and max ingested char_end. Largest gap: %d (%s). "
            "Run tests/eval/INGEST_HANDOFF.md remediation.",
            len(truncated), threshold,
            truncated[0]["gap"], truncated[0]["title"][:60],
        )
        for t in truncated[:5]:
            log.warning(
                "  gap=%6d  qdrant_max=%7d  src_len=%7d  %s",
                t["gap"], t["qdrant_max_char_end"], t["source_length"], t["title"][:70],
            )
        if len(truncated) > 5:
            log.warning("  ... and %d more — see truncation_report.json", len(truncated) - 5)
    else:
        log.info("Truncation check passed: all ingested titles cover their full source text.")
    return truncated


def sample_doc_scoped(points: list[dict[str, Any]]) -> list[str]:
    titles = sorted({p["title"] for p in points})
    rng = random.Random(RNG_SEED)
    sample = rng.sample(titles, min(DOC_SAMPLE_SIZE, len(titles)))
    log.info("Doc-scoped sample (seed=%d): %d titles", RNG_SEED, len(sample))
    return sample


def summary_stats(name: str, sizes: list[int]) -> None:
    if not sizes:
        log.info("%s: empty", name)
        return
    sizes_sorted = sorted(sizes)
    median = sizes_sorted[len(sizes_sorted) // 2]
    log.info(
        "%s |G|: min=%d median=%d max=%d (n=%d, %d zero)",
        name,
        min(sizes),
        median,
        max(sizes),
        len(sizes),
        sum(1 for s in sizes if s == 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-chunks", action="store_true",
                    help="Leave gold.json and doc_scoped_titles.json untouched.")
    ap.add_argument(
        "--collection",
        default=COLLECTION,
        help="Qdrant collection to scroll (default: QDRANT_COLLECTION env var or 'cuad_contracts').",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory to write gold.json / gold_contracts.json / gold_spans.json / "
             "doc_scoped_titles.json (default: tests/eval/).",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_out          = out_dir / "gold.json"
    gold_contracts_out = out_dir / "gold_contracts.json"
    gold_spans_out    = out_dir / "gold_spans.json"
    doc_scoped_out    = out_dir / "doc_scoped_titles.json"

    log.info("Collection: %s  |  out_dir: %s", args.collection, out_dir)
    queries = json.loads(QUERIES_PATH.read_text())
    spans, source_lengths = parse_spans(fetch_cuad_json())
    points = scroll_points(collection_name=args.collection)
    if not points:
        log.error("Qdrant returned 0 points; refusing to write empty gold.")
        return 2
    ingested_titles = {p["title"] for p in points}
    log.info("Ingested titles in Qdrant: %d", len(ingested_titles))

    # --- Truncation check: run before building gold so gaps are visible upfront ---
    truncated = check_truncation(points, source_lengths)
    if truncated:
        truncation_out = out_dir / "truncation_report.json"
        truncation_out.write_text(json.dumps(truncated, indent=2))
        log.warning("Full truncation report written to %s", truncation_out)

    if not args.skip_chunks:
        gold = build_gold_chunks(queries, spans, points)
        gold_out.write_text(json.dumps(gold, indent=2))
        log.info("Wrote %s with %d entries", gold_out, len(gold))
        summary_stats("gold.json (chunks)", [len(v) for v in gold.values()])
        doc_scoped_out.write_text(json.dumps(sample_doc_scoped(points), indent=2))
        log.info("Wrote %s", doc_scoped_out)
    else:
        log.info("--skip-chunks: gold.json + doc_scoped_titles.json untouched.")

    gc = build_gold_contracts(queries, spans, ingested_titles)
    gold_contracts_out.write_text(json.dumps(gc, indent=2))
    log.info("Wrote %s with %d entries", gold_contracts_out, len(gc))
    summary_stats("gold_contracts.json (docs)", [v["total_relevant"] for v in gc.values()])

    gs = build_gold_spans(queries, spans, ingested_titles)
    gold_spans_out.write_text(json.dumps(gs, indent=2))
    total = sum(sum(len(v) for v in e["spans_by_document"].values()) for e in gs.values())
    largest = max(
        (len(v) for e in gs.values() for v in e["spans_by_document"].values()),
        default=0,
    )
    log.info("Wrote %s; %d entries, %d spans total, largest (qid,doc)=%d",
             gold_spans_out, len(gs), total, largest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
