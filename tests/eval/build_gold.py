"""
build_gold.py
─────────────
Build the binary-relevance gold set for CUAD retrieval evaluation.

Pipeline:
  1. Download CUAD_v1.json from the HuggingFace dataset repo
     `theatticusproject/cuad` (HF-only by user decision; no GitHub fallback).
  2. Parse it into (title, category, answer_start, answer_end) spans.
  3. Scroll the Qdrant `cuad_contracts` collection and collect every point's
     (id, title, char_start, char_end).
  4. For each query id in queries.json, emit the set of point_ids whose
     char interval overlaps any gold span of the query's category by ≥1 char.
  5. Sample 5 documents (stratified by CUAD Part directory when available,
     else uniform) with seed=42 for the doc-scoped subset.

Outputs:
  tests/eval/gold.json              # { query_id: [point_id, ...] }
  tests/eval/doc_scoped_titles.json # [title, ...] (5 titles)

Run:
  python tests/eval/build_gold.py
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

# Make the cuad-demo-quadrant package importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cuad-demo-quadrant"))

from huggingface_hub import hf_hub_download  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from qdrant_cluster_connect import get_qdrant_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_gold")

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / "cuad_gold" / "_hf_cache"
QUERIES_PATH = EVAL_DIR / "queries.json"
GOLD_OUT = EVAL_DIR / "gold.json"
DOC_SCOPED_OUT = EVAL_DIR / "doc_scoped_titles.json"
COLLECTION = "cuad_contracts"
DOC_SAMPLE_SIZE = 5
RNG_SEED = 42

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


def parse_spans(cuad_json_path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Return {(title, category): [(answer_start, answer_end), ...]}."""
    with cuad_json_path.open() as f:
        data = json.load(f)
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for doc in data["data"]:
        title = doc["title"].strip()
        for para in doc["paragraphs"]:
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
                    out.setdefault((title, cat), []).append((int(start), int(start) + len(text)))
    log.info("Parsed %d (title, category) span groups", len(out))
    return out


def scroll_points() -> list[dict[str, Any]]:
    """Return every point's id + title/char_start/char_end."""
    client = get_qdrant_client()
    points: list[dict[str, Any]] = []
    offset = None
    page = 0
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=["title", "char_start", "char_end"],
            with_vectors=False,
        )
        for p in batch:
            pl = p.payload or {}
            title = (pl.get("title") or "").strip()
            cs = pl.get("char_start")
            ce = pl.get("char_end")
            if not title or cs is None or ce is None:
                continue
            points.append({"id": p.id, "title": title, "char_start": int(cs), "char_end": int(ce)})
        page += 1
        if offset is None:
            break
    log.info("Scrolled %d points across %d pages", len(points), page)
    return points


def build_gold(
    queries: list[dict[str, Any]],
    spans: dict[tuple[str, str], list[tuple[int, int]]],
    points: list[dict[str, Any]],
) -> dict[str, list]:
    """For each query, return point_ids whose char interval overlaps a span."""
    # Index points by title for fast lookup.
    by_title: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_title.setdefault(p["title"], []).append(p)

    gold: dict[str, list] = {}
    for q in queries:
        qid = q["id"]
        cat = q["category"]
        hits: list = []
        # Walk every title that has a span of this category.
        for (title, c), span_list in spans.items():
            if c != cat:
                continue
            for p in by_title.get(title, []):
                pcs, pce = p["char_start"], p["char_end"]
                if any(pcs < ae and ace_start < pce for ace_start, ae in span_list):
                    hits.append(p["id"])
                    # don't break — a point might be hit by multiple spans,
                    # but we only record it once below.
        # Dedupe preserving order.
        seen = set()
        deduped = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                deduped.append(h)
        gold[qid] = deduped
    return gold


def sample_doc_scoped(points: list[dict[str, Any]]) -> list[str]:
    """Stratified-by-CUAD-Part sample of DOC_SAMPLE_SIZE titles, seed=42.

    The CUAD repo lays PDFs out as `CUAD_v1/full_contract_pdf/Part_<I|II|III>/<Type>/<Title>.pdf`,
    but the Qdrant payload only carries `title`. We don't have part/type
    cheaply here, so we do a deterministic uniform sample over sorted titles.
    """
    titles = sorted({p["title"] for p in points})
    rng = random.Random(RNG_SEED)
    sample = rng.sample(titles, min(DOC_SAMPLE_SIZE, len(titles)))
    log.info("Doc-scoped sample (seed=%d): %d titles", RNG_SEED, len(sample))
    return sample


def main() -> int:
    queries = json.loads(QUERIES_PATH.read_text())
    cuad_path = fetch_cuad_json()
    spans = parse_spans(cuad_path)
    points = scroll_points()
    if not points:
        log.error("Qdrant returned 0 points; refusing to write empty gold.")
        return 2
    gold = build_gold(queries, spans, points)
    GOLD_OUT.write_text(json.dumps(gold, indent=2))
    log.info("Wrote %s with %d query entries", GOLD_OUT, len(gold))
    sample = sample_doc_scoped(points)
    DOC_SCOPED_OUT.write_text(json.dumps(sample, indent=2))
    log.info("Wrote %s", DOC_SCOPED_OUT)
    # Summary stats.
    sizes = [len(v) for v in gold.values()]
    sizes_sorted = sorted(sizes)
    if sizes:
        median = sizes_sorted[len(sizes_sorted) // 2]
    else:
        median = 0
    log.info(
        "Gold |G(Q)|: min=%d median=%d max=%d (n=%d, %d queries with 0 relevant)",
        min(sizes) if sizes else 0,
        median,
        max(sizes) if sizes else 0,
        len(sizes),
        sum(1 for s in sizes if s == 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
