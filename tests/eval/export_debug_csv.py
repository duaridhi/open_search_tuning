"""Emit a debug CSV joining gold.json + gold_spans.json + Qdrant point payloads.

One row per (query_id, point_id, overlapping_span). Each row carries the chunk
text, the labeled CUAD span text, character-range overlap, and a direct Qdrant
point URL for inspection.

Run: python tests/eval/export_debug_csv.py
Output: tests/eval/gold_debug.csv
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "cuad-demo-quadrant"))
from qdrant_cluster_connect import get_qdrant_client  # noqa: E402

ROOT = Path(__file__).resolve().parent
COLLECTION = "cuad_contracts"
QDRANT_BASE = "https://a780cbd4-ea1e-43f5-a54a-c73f0d485a70.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_POINT_URL = f"{QDRANT_BASE}/collections/{COLLECTION}/points/{{point_id}}"
TEXT_CAP = 400


def truncate(s: str | None, n: int = TEXT_CAP) -> str:
    if not s:
        return ""
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def load_point_index(needed_ids: set[str]) -> dict[str, dict]:
    """Scroll the whole collection once, keep only points we need."""
    client = get_qdrant_client()
    index: dict[str, dict] = {}
    offset = None
    fields = ["title", "text", "char_start", "char_end"]
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=fields,
            limit=512,
            offset=offset,
        )
        for p in pts:
            pid = str(p.id)
            if pid in needed_ids:
                index[pid] = p.payload
        if offset is None:
            break
    return index


def overlapping_spans(chunk_start: int, chunk_end: int, spans: list[dict]) -> list[dict]:
    return [s for s in spans if s["char_start"] < chunk_end and s["char_end"] > chunk_start]


def main() -> None:
    queries = {q["id"]: q for q in json.loads((ROOT / "queries.json").read_text())}
    gold = json.loads((ROOT / "gold.json").read_text())
    gold_spans = json.loads((ROOT / "gold_spans.json").read_text())

    needed_ids = {pid for pids in gold.values() for pid in pids}
    print(f"Loading payload for {len(needed_ids)} unique point_ids from Qdrant...")
    point_index = load_point_index(needed_ids)
    print(f"  resolved {len(point_index)}/{len(needed_ids)} points")

    out_path = ROOT / "gold_debug.csv"
    cols = [
        "query_id", "category", "form", "query_text",
        "title",
        "point_id", "qdrant_url",
        "chunk_char_start", "chunk_char_end", "chunk_text",
        "span_char_start", "span_char_end", "span_text",
        "overlap_chars",
    ]
    rows_written = 0
    missing_payload = 0
    missing_span_match = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for qid, point_ids in gold.items():
            q = queries.get(qid, {})
            spans_by_doc = gold_spans.get(qid, {}).get("spans_by_document", {})
            for pid in point_ids:
                pl = point_index.get(pid)
                if not pl:
                    missing_payload += 1
                    continue
                title = pl.get("title", "")
                c_start = pl.get("char_start") or 0
                c_end = pl.get("char_end") or 0
                spans = spans_by_doc.get(title, [])
                hits = overlapping_spans(c_start, c_end, spans)
                base_row = {
                    "query_id": qid,
                    "category": q.get("category", ""),
                    "form": q.get("form", ""),
                    "query_text": truncate(q.get("q", "")),
                    "title": title,
                    "point_id": pid,
                    "qdrant_url": QDRANT_POINT_URL.format(point_id=pid),
                    "chunk_char_start": c_start,
                    "chunk_char_end": c_end,
                    "chunk_text": truncate(pl.get("text", "")),
                }
                if not hits:
                    missing_span_match += 1
                    w.writerow({**base_row, "span_char_start": "", "span_char_end": "",
                                "span_text": "", "overlap_chars": 0})
                    rows_written += 1
                    continue
                for s in hits:
                    overlap = max(0, min(c_end, s["char_end"]) - max(c_start, s["char_start"]))
                    w.writerow({
                        **base_row,
                        "span_char_start": s["char_start"],
                        "span_char_end": s["char_end"],
                        "span_text": truncate(s.get("answer_text", "")),
                        "overlap_chars": overlap,
                    })
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {out_path}")
    print(f"  missing payload (point not in Qdrant): {missing_payload}")
    print(f"  no overlapping span found:             {missing_span_match}")


if __name__ == "__main__":
    main()
