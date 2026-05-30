"""
analyze_drift.py
────────────────
Measures char-offset drift between CUAD's TXT-derived context strings and the
text produced by pdfplumber PDF extraction for the same contracts.

Usage
─────
    # Analyse one document and print a detailed report:
    python tests/eval/cuad_txt_vs_pdf_extract_analysis/analyze_drift.py \
        --title "TRANSMONTAIGNEPARTNERSLLC_03_13_2020-EX-10.9-SERVICES AGREEMENT"

    # Scan every contract in CUAD and rank by total drift (slow — ~510 PDFs):
    python tests/eval/cuad_txt_vs_pdf_extract_analysis/analyze_drift.py --scan-all

    # Override PDF root or CUAD json path:
    python ... --pdf-root /path/to/full_contract_pdf \
               --cuad-json /path/to/CUAD_v1.json

Environment
───────────
    PDF_ROOT   override for the full_contract_pdf directory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# ── defaults ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent

DEFAULT_CUAD_JSON = (
    _EVAL_DIR
    / "cuad_gold"
    / "_hf_cache"
    / "datasets--theatticusproject--cuad"
    / "snapshots"
    / "a3c393f5d103fd0c516374e4fdff676c8176dcb1"
    / "CUAD_v1"
    / "CUAD_v1.json"
)

_DEFAULT_PDF_ROOT = Path(
    os.getenv(
        "PDF_ROOT",
        "/home/ridhi/projects/project1/open_search_tuning"
        "/cuad_opensearch/cuad_data/CUAD_v1/full_contract_pdf",
    )
)

# ── text extraction ───────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except ImportError:
        pass
    try:
        import fitz
        pages = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                pages.append(page.get_text("text") or "")
        return "\n\n".join(pages)
    except ImportError:
        pass
    raise RuntimeError("Install pdfplumber or pymupdf (fitz) to extract PDF text.")


def find_pdf(title: str, pdf_root: Path) -> Path | None:
    for p in pdf_root.rglob("*.pdf"):
        if p.stem.upper() == title.upper():
            return p
    for p in pdf_root.rglob("*.PDF"):
        if p.stem.upper() == title.upper():
            return p
    return None


# ── drift classification ──────────────────────────────────────────────────────

def classify_drift(cuad_text: str, pdf_text: str) -> dict:
    """Run SequenceMatcher and bucket every non-equal block."""
    sm = SequenceMatcher(None, cuad_text, pdf_text, autojunk=False)
    opcodes = sm.get_opcodes()

    extra_nl_in_cuad = 0   # cause 1: drift-accumulating
    space_nl_swaps   = 0   # cause 2: no drift but breaks search
    quote_swaps      = 0   # cause 3: no drift but breaks search
    other            = 0

    _CURLY = str.maketrans({0x201C: chr(34), 0x201D: chr(34), 0x2018: chr(39), 0x2019: chr(39)})

    extra_nl_events: list[dict] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        c = cuad_text[i1:i2]
        p = pdf_text[j1:j2]

        if tag == "delete":
            if all(ch == "\n" for ch in c):
                extra_nl_in_cuad += len(c)
                extra_nl_events.append({
                    "cuad_pos": i1,
                    "count": len(c),
                    "context": cuad_text[max(0, i1 - 30): i1]
                               + ">>>" + repr(c) + "<<<"
                               + cuad_text[i2: i2 + 30],
                })
            else:
                other += 1
        elif tag == "insert":
            other += 1
        elif tag == "replace":
            c_norm = re.sub(r"\s+", " ", c.translate(_CURLY)).strip()
            p_norm = re.sub(r"\s+", " ", p.translate(_CURLY)).strip()
            if c_norm == p_norm:
                if set(c + p) <= {" ", "\n"}:
                    space_nl_swaps += 1
                else:
                    quote_swaps += 1
            else:
                other += 1

    net_drift = len(pdf_text) - len(cuad_text)

    return {
        "cuad_len": len(cuad_text),
        "pdf_len": len(pdf_text),
        "net_drift": net_drift,
        "extra_nl_in_cuad": extra_nl_in_cuad,
        "space_nl_swaps": space_nl_swaps,
        "quote_swaps": quote_swaps,
        "other_blocks": other,
        "extra_nl_events": extra_nl_events,
    }


def anchor_drift(cuad_text: str, pdf_text: str) -> list[dict]:
    """Return measured drift at a handful of anchor phrases."""
    anchors = [
        "AGREEMENT is entered",
        "Section 1",
        "Section 2",
        "Article I",
        "Article II",
        "governed by",
        "Governing Law",
        "Signature Page",
        "IN WITNESS WHEREOF",
    ]
    rows = []
    for phrase in anchors:
        c = cuad_text.lower().find(phrase.lower())
        p = pdf_text.lower().find(phrase.lower())
        if c == -1 or p == -1:
            continue
        rows.append({"phrase": phrase, "cuad_pos": c, "pdf_pos": p, "drift": p - c})
    return rows


# ── single-document report ────────────────────────────────────────────────────

def report_one(title: str, cuad_json: Path, pdf_root: Path) -> None:
    with cuad_json.open() as f:
        data = json.load(f)

    cuad_text = None
    for doc in data["data"]:
        if doc["title"].strip() == title:
            cuad_text = doc["paragraphs"][0]["context"]
            break
    if cuad_text is None:
        print(f"Title not found in CUAD: {title}")
        sys.exit(1)

    pdf_path = find_pdf(title, pdf_root)
    if pdf_path is None:
        print(f"PDF not found under {pdf_root} for title: {title}")
        sys.exit(1)

    pdf_text = extract_pdf_text(pdf_path)
    result = classify_drift(cuad_text, pdf_text)
    anchors = anchor_drift(cuad_text, pdf_text)

    print(f"\n{'='*70}")
    print(f"Document: {title}")
    print(f"{'='*70}")
    print(f"  CUAD context length : {result['cuad_len']:,}")
    print(f"  PDF text length     : {result['pdf_len']:,}")
    print(f"  Net drift (PDF-CUAD): {result['net_drift']:+d}")
    print()
    print(f"Cause 1 — extra \\n in CUAD (cumulative drift): {result['extra_nl_in_cuad']} chars")
    print(f"Cause 2 — space↔newline swaps (no drift):      {result['space_nl_swaps']}")
    print(f"Cause 3 — curly↔ASCII quote swaps (no drift):  {result['quote_swaps']}")
    print(f"Other diff blocks:                              {result['other_blocks']}")

    if anchors:
        print(f"\nAnchor-phrase drift table:")
        print(f"  {'Phrase':30s}  {'CUAD':>7}  {'PDF':>7}  {'Drift':>6}")
        print(f"  {'-'*57}")
        for row in anchors:
            print(f"  {row['phrase']:30s}  {row['cuad_pos']:>7}  {row['pdf_pos']:>7}  {row['drift']:>+6}")

    if result["extra_nl_events"]:
        print(f"\nExtra-newline events (Cause 1 detail):")
        for ev in result["extra_nl_events"]:
            print(f"  pos={ev['cuad_pos']:5d} ({ev['count']} nl): {ev['context']}")


# ── scan-all mode ─────────────────────────────────────────────────────────────

def scan_all(cuad_json: Path, pdf_root: Path) -> None:
    with cuad_json.open() as f:
        data = json.load(f)

    rows = []
    total = len(data["data"])
    for i, doc in enumerate(data["data"], 1):
        title = doc["title"].strip()
        cuad_text = doc["paragraphs"][0]["context"]
        pdf_path = find_pdf(title, pdf_root)
        if pdf_path is None:
            print(f"[{i}/{total}] SKIP (no PDF): {title[:60]}")
            continue
        try:
            pdf_text = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"[{i}/{total}] ERROR: {title[:60]}: {exc}")
            continue
        result = classify_drift(cuad_text, pdf_text)
        rows.append({"title": title, **result})
        if i % 50 == 0:
            print(f"[{i}/{total}] processed …")

    rows.sort(key=lambda r: abs(r["net_drift"]), reverse=True)
    print(f"\n{'='*80}")
    print(f"Scan complete — {len(rows)} contracts analysed")
    print(f"{'='*80}")
    print(f"{'Title':60s}  {'drift':>6}  {'extra_nl':>8}  {'nl_swaps':>8}")
    print(f"{'-'*80}")
    for r in rows[:40]:
        print(
            f"{r['title'][:60]:60s}  {r['net_drift']:>+6}  "
            f"{r['extra_nl_in_cuad']:>8}  {r['space_nl_swaps']:>8}"
        )
    if len(rows) > 40:
        print(f"… and {len(rows) - 40} more")

    # Summary stats
    drifts = [abs(r["net_drift"]) for r in rows]
    extra_nls = [r["extra_nl_in_cuad"] for r in rows]
    print(f"\nDrift |abs| — min={min(drifts)} median={sorted(drifts)[len(drifts)//2]} max={max(drifts)}")
    print(f"Extra NL     — min={min(extra_nls)} median={sorted(extra_nls)[len(extra_nls)//2]} max={max(extra_nls)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", help="Exact CUAD document title to analyse")
    ap.add_argument("--scan-all", action="store_true", help="Analyse every contract")
    ap.add_argument("--cuad-json", type=Path, default=DEFAULT_CUAD_JSON)
    ap.add_argument("--pdf-root", type=Path, default=_DEFAULT_PDF_ROOT)
    args = ap.parse_args()

    if not args.cuad_json.exists():
        print(f"CUAD JSON not found: {args.cuad_json}")
        print("Run `python tests/eval/build_gold.py` once to download it.")
        sys.exit(1)

    if not args.pdf_root.exists():
        print(f"PDF root not found: {args.pdf_root}")
        print("Set --pdf-root or the PDF_ROOT env var.")
        sys.exit(1)

    if args.scan_all:
        scan_all(args.cuad_json, args.pdf_root)
    elif args.title:
        report_one(args.title, args.cuad_json, args.pdf_root)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
