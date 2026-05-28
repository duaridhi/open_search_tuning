# Chunking Strategy

## Summary

The ingestion pipeline ([upload_to_qdrant.py](../cuad-demo-quadrant/upload_to_qdrant.py)) uses **fixed-size character chunking with overlap and natural boundary snapping**.

Each PDF is extracted page-by-page, pages are joined with `\n\n`, and the resulting full-document string is split by `split_text_with_offsets()` using these parameters (all tunable via env vars):

| Parameter | Default | Env var |
|---|---|---|
| Chunk size | 500 chars | `CHUNK_SIZE` |
| Overlap | 50 chars | `CHUNK_OVERLAP` |

**Boundary snapping**: before finalising each chunk boundary, the splitter looks backwards for the nearest natural separator in priority order: `\n\n` → `\n` → ` ` → (hard cut). A separator is only accepted if it falls in the second half of the window (`idx > chunk_size // 2`), preventing degenerate single-word chunks.

**Overlap**: the next chunk starts `chunk_overlap` characters before the end of the previous chunk, so clause text that straddles a boundary appears in both adjacent chunks.

**Metadata preserved per chunk**: `char_start`, `char_end`, `page_start`, `page_end`, `page_offset_start`, `page_offset_end`, `title`, `pdf_path`, `doc_id` (`{title}-chunk-{idx}`).

Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine, L2-normalised) and upserted into Qdrant with a deterministic UUID seeded from `doc_id`, making re-ingestion idempotent.

---

## Pros

- **Simple and predictable.** Every chunk is close to the same character length, so embedding quality and Qdrant payload sizes are uniform and easy to reason about.
- **Overlap prevents hard clause splits.** The 50-char overlap means a key phrase that falls at a boundary will appear in at least one chunk in full, improving recall for short queries.
- **Natural boundary snapping reduces mid-word cuts.** Preferring `\n\n` and `\n` over hard character cuts keeps most chunks ending on a paragraph or line boundary, preserving readability in highlights.
- **Fully idempotent.** Deterministic UUIDs mean re-running the pipeline replaces rather than duplicates points; safe to re-run after parameter tweaks.
- **Tunable without code changes.** `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MAX_DOCS` are all env vars, so experimenting with different granularities requires no edits.
- **Page-level provenance.** Each chunk records `page_start`/`page_end` and intra-page character offsets, enabling the API to link results back to the exact page in the source PDF.

---

## Cons

- **Character count ≠ token count.** The 500-char window does not account for the model's 256-token max sequence length (`all-MiniLM-L6-v2`). Dense legal prose (~4 chars/token) fits within ~125 tokens — well under the cap — but formatted tables or lists with many short tokens could silently exceed it and be truncated by the tokeniser, producing degraded embeddings without any warning.
- **Ignores semantic structure.** The splitter has no knowledge of contract sections, article headings, or clause boundaries. A clause that spans 800 characters will be split in the middle, and both halves lose context about which section they belong to.
- **No sentence-aware boundaries.** The fallback separators (`\n\n`, `\n`, ` `) do not align to sentence endings. A chunk may start or end mid-sentence if no whitespace falls in the right part of the window.
- **Overlap is small relative to chunk size (10%).** A 50-char overlap carries roughly one short phrase. If the relevant clause begins 100+ chars before a boundary it will still be split across non-overlapping chunks.
- **Full-document text join loses page structure.** Pages are concatenated with `\n\n` before chunking, so a chunk can silently span a page boundary. The page metadata is recovered by a separate character-offset lookup — an approximation that can assign a chunk to the wrong page if the extracted text has irregular whitespace.
- **No deduplication of repeated boilerplate.** CUAD contracts often contain identical header/footer text on every page. These repeating strings generate many near-identical chunks that waste index capacity and dilute search results without adding retrieval value.
- **Slow extraction fallback.** Without `pymupdf`, the pipeline falls back to `pdfplumber`, which is significantly slower on 510 PDFs and can mis-extract text from PDFs with complex layouts (tables, multi-column text).
