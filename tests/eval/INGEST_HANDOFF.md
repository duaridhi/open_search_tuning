# INGEST_HANDOFF.md — TXT/PDF Drift in Truncation Check

## What the warning means

`build_gold.py` compares two char-position systems:

| Source | What it measures |
|--------|-----------------|
| `source_length` | Length of the CUAD `.txt` annotation context (from `CUAD_v1.json`) |
| `qdrant_max_char_end` | Largest `char_end` among ingested PDF chunks for that title |

When `source_length − qdrant_max_char_end > 200`, a "TXT/PDF DRIFT" warning fires.

## Why it is not a bug

CUAD annotations were made against `.txt` files extracted from PDFs years ago.
The ingest pipeline (`upload_to_qdrant_hf.py`) re-extracts text directly from the
original PDFs using `pymupdf`. PDF extraction strips:

- page headers and footers
- page numbers
- table-of-contents lines
- blank separator pages

This produces shorter text, so `qdrant_max_char_end < source_length` for almost
every document. The effect is consistent: **~36 of 50 ingested docs** show this
gap in every collection, regardless of embedding model or chunk size.

## Impact on eval

| Gold artifact | Built with | Affected? |
|---------------|-----------|-----------|
| `gold.json` (chunk-level) | Text substring matching | **No** — matches answer text directly, not by char offset |
| `gold_contracts.json` (contract-level) | Title membership | **No** |
| `gold_spans.json` (span-level) | CUAD TXT char offsets | **Yes** — spans in the tail may not align to PDF chunks |

The primary eval metrics (Recall@k, MRR, contract recall) all use `gold.json` /
`gold_contracts.json`. Span-level alignment in `gold_spans.json` is a secondary
diagnostic and its accuracy degrades for the tail region.

## Remediation (none needed for current eval)

The current eval pipeline is unaffected. If span-level eval accuracy becomes
important, the fix is to re-derive char offsets by running a text-search over the
PDF-extracted full document text to locate each answer span, replacing the
TXT-derived `char_start`/`char_end` with PDF-derived positions. That work is
tracked separately.
