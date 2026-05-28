---
name: eval-dataset
description: Pre-ingest evaluation of a new PDF dataset. Samples the PDFs, reports size/character-count distribution, estimates total chunk count under the current chunking parameters (CHUNK_SIZE / CHUNK_OVERLAP), flags extraction warnings (empty pages, oversized files, pdfplumber vs pymupdf fallback), and identifies scanned/image-only PDFs that require OCR before they can be ingested. Use before running upload_to_qdrant.py on a new corpus so you know what to expect.
---

# Dataset pre-ingest evaluation

Runs `tests/eval/eval_dataset.py` against a PDF directory and reports stats.

## When to invoke

- "How many chunks will this produce?"
- "Evaluate / profile the new PDF dataset"
- "What's the chunk estimate for this corpus?"
- Any explicit `/eval-dataset` invocation.

Do NOT invoke for: running the actual ingestion (`cuad-ingest` agent owns that), search quality checks (`rag-eval`), or latency profiling (`/perf`).

## Command

```bash
python tests/eval/eval_dataset.py [PDF_ROOT] [options]
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `PDF_ROOT` (positional) | path in `upload_to_qdrant.py` | Directory containing PDFs |
| `--chunk-size` | 500 | Characters per chunk |
| `--chunk-overlap` | 50 | Overlap between chunks |
| `--step` | 10 | Sample every Nth PDF (use `1` for full scan) |
| `--ocr-threshold` | 50 | Chars/page below which a PDF is flagged as OCR candidate |

If the user doesn't provide `PDF_ROOT`, omit it — the script defaults to the path hardcoded in `upload_to_qdrant.py`.

## Steps

1. Run the command with any args the user provided.
2. Read stdout and report the summary table to the user.
3. Flag any warnings printed by the script:
   - **Empty PDFs** — no text layer; must run `ocrmypdf input.pdf output.pdf` before ingest.
   - **Sparse-text PDFs** — < 50 chars/page; likely scanned with poor OCR; `ocrmypdf --force-ocr` recommended.
   - **Extraction errors** — investigate before ingesting; these will be skipped with a WARNING log.
   - **pdfplumber fallback** — suggest installing `pymupdf` for faster, more accurate extraction.
4. If OCR candidates exist, suggest the remediation command:
   ```bash
   # re-OCR a single file
   ocrmypdf --force-ocr input.pdf input.pdf

   # batch re-OCR all candidates
   for f in <candidate_list>; do ocrmypdf --force-ocr "$f" "$f"; done
   ```
5. Exit code `1` means issues were found (OCR candidates or errors); exit code `0` means the corpus is clean.

## What this skill does NOT do

- It does not run the ingestion. Hand off to the `cuad-ingest` agent for that.
- It does not check the Qdrant collection or vector schema. The `qdrant-payload` agent owns that.
- It does not evaluate search quality. The `rag-eval` agent owns that.

## File map

| Path | Role |
|---|---|
| `tests/eval/eval_dataset.py` | The script this skill runs |
| `cuad-demo-quadrant/upload_to_qdrant.py` | Source of default `PDF_ROOT`, `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| `readme_docs/CHUNKING_STRATEGY.md` | Chunking strategy docs and pros/cons |
