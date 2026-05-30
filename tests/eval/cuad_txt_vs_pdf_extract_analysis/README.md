# CUAD TXT vs PDF Extraction — Char Offset Drift Analysis

## Background

The CUAD evaluation gold data (`CUAD_v1.json`) contains expert-annotated clause
spans expressed as `(answer_start, answer_text)` pairs referencing positions in
each contract's `context` string.  That `context` string originates from the
`full_contract_txt/` plain-text files shipped alongside the PDFs.

The search service ingests the same contracts from their PDFs via pymupdf /
pdfplumber.  The two text representations are **not byte-identical**, so CUAD's
`answer_start` char positions do not map directly to the `char_start`/`char_end`
values stored in Qdrant.

## Three causes of divergence

### Cause 1 — Extra blank lines in TXT (drift-accumulating) ★

The TXT files have additional `\n` characters around:

- **Section headers** (`ARTICLE I`, `ARTICLE II`, …) — one extra blank line before each
- **Page numbers** (`1`, `2`, `3` as standalone tokens) — 4–6 extra newlines surrounding each

The PDF extraction either omits these tokens entirely or renders them inline with
much less surrounding whitespace.  Every extra `\n` in TXT shifts all subsequent
TXT char positions by +1 relative to the PDF positions.  The shift **accumulates**
— it is not constant and cannot be corrected with a single offset.

**Measured on TRANSMONTAIGNE Services Agreement (5,886-char document):**

| Anchor phrase        | CUAD pos | PDF pos | Drift |
|---|---|---|---|
| SERVICES AGREEMENT…  | 39       | 38      | −1    |
| Section 1            | 2,477    | 2,460   | −17   |
| "governed by"        | 2,896    | 2,876   | −20   |
| Signature Page       | 5,525    | 5,493   | −32   |
| End of document      | 5,886    | 5,848   | −38   |

Total: 25 extra-newline events accounting for 34 extra chars in TXT.

### Cause 2 — Intra-paragraph line wraps (no drift, breaks text search)

PDF rendering preserves physical line boundaries within flowing paragraphs:

```
PDF:  "governed by the\nlaws of the State of Colorado"
CUAD: "governed by the laws of the State of Colorado"
```

Both are a 1-for-1 character substitution (`\n` ↔ space), so cumulative drift
does not increase, but **exact substring search fails** for any phrase that
crosses a line boundary.

Measured: **49 such swaps** in the 5,886-char example contract.

### Cause 3 — Unicode smart quotes vs ASCII (no drift, breaks quoted-phrase search)

The TXT files retain smart/curly quotes from the original word-processor source:

| TXT (CUAD) | PDF extraction |
|---|---|
| `"` `"` (U+201C / U+201D) | `"` (U+0022) |
| `'` `'` (U+2018 / U+2019) | `'` (U+0027) |

Also a 1-for-1 substitution (no drift), but breaks searches for quoted terms.

Measured: **19 quote-substitution blocks** in the example contract.

## Fix applied to the eval pipeline

Both `build_gold.py` (`build_gold_chunks`) and `run_eval.py` (`_compute_span_hit`)
were updated to use **normalised text substring matching** instead of char-offset
numeric overlap:

```python
_CURLY_TABLE = str.maketrans('""''', '""\'\'')

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.translate(_CURLY_TABLE)).strip()

# Match: _norm(answer_text) in _norm(chunk_text)
```

This collapses all whitespace variants to a single space (fixes Causes 1 & 2)
and folds curly quotes to ASCII (fixes Cause 3).  Spans shorter than 20 chars
are skipped to avoid false-positive matches.

`char_start`/`char_end` in the Qdrant payload and search API response are
**unchanged** — they are still used by the UI to locate and highlight text inside
the rendered PDF.

## How to re-run the analysis

```bash
# Anchor-phrase drift for one document:
python tests/eval/cuad_txt_vs_pdf_extract_analysis/analyze_drift.py \
    --title "TRANSMONTAIGNEPARTNERSLLC_03_13_2020-EX-10.9-SERVICES AGREEMENT"

# Different document:
python tests/eval/cuad_txt_vs_pdf_extract_analysis/analyze_drift.py \
    --title "YOUR_CONTRACT_TITLE_HERE"

# Scan all contracts and rank by total drift:
python tests/eval/cuad_txt_vs_pdf_extract_analysis/analyze_drift.py --scan-all
```

## Files

| File | Purpose |
|---|---|
| `README.md` | This document |
| `analyze_drift.py` | Reusable drift analysis script |
