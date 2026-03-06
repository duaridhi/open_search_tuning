"""
extract_index_cuad_pdfs.py
──────────────────────────
Indexes CUAD contracts sourced directly from the PDF files instead of
the pre-extracted CUAD_v1.json.

Key differences vs 02_index_cuad_documents.py
──────────────────────────────────────────────
• Text is extracted from every PDF in full_contract_pdf/ recursively.
• Each indexed chunk carries two extra fields:
    - page_start  (int) – first PDF page the chunk overlaps
    - page_end    (int) – last  PDF page the chunk overlaps
• The index is separate (cuad_pdf_dataset by default) so it can coexist
  with the JSON-derived cuad_dataset index.
• PDF extraction tries pdfplumber first; falls back to pdfminer.six.
  Install pymupdf (pip install pymupdf) for faster / higher-quality
  extraction — the code detects and prefers it automatically.

Usage
─────
    # Full run
    python extract_index_cuad_pdfs.py

    # Limit to first N PDFs (useful for testing)
    MAX_PDFS=5 python extract_index_cuad_pdfs.py

    # Point at a different OpenSearch index
    
    my_index python extract_index_cuad_pdfs.py
"""

# %% Imports & environment
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from opensearchpy import helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# %% PDF extraction backend — prefer pymupdf > pdfplumber > pdfminer
def _make_extractor():
    """
    Return (name, extract_fn) where extract_fn(path) -> list[{"page": int, "text": str}].
    Tries pymupdf → pdfplumber → pdfminer in order.
    """
    # ── pymupdf (fastest, best layout) ──────────────────────────────────
    try:
        import fitz  # pymupdf

        def _extract_pymupdf(path: Path):
            pages = []
            with fitz.open(str(path)) as doc:
                for i, page in enumerate(doc):
                    pages.append({"page": i + 1, "text": page.get_text("text") or ""})
            return pages

        return "pymupdf", _extract_pymupdf
    except ImportError:
        pass

    # ── pdfplumber (installed) ───────────────────────────────────────────
    try:
        import pdfplumber

        def _extract_pdfplumber(path: Path):
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    pages.append({"page": i + 1, "text": page.extract_text() or ""})
            return pages

        return "pdfplumber", _extract_pdfplumber
    except ImportError:
        pass

    # ── pdfminer.six (fallback) ──────────────────────────────────────────
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTAnon, LTChar, LTTextContainer

        def _extract_pdfminer(path: Path):
            pages = []
            for i, page_layout in enumerate(extract_pages(str(path))):
                text_parts = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text_parts.append(element.get_text())
                pages.append({"page": i + 1, "text": "".join(text_parts)})
            return pages

        return "pdfminer", _extract_pdfminer
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF extraction library found. "
        "Install one of: pymupdf, pdfplumber, pdfminer.six"
    )


EXTRACTOR_NAME, extract_pages_from_pdf = _make_extractor()
print(f"PDF extractor : {EXTRACTOR_NAME}")


# %% Text-splitting (same algorithm as 02_index_cuad_documents.py)
def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Split *text* into chunks of ~chunk_size chars with overlap.
    Returns list of {"text", "char_start", "char_end"}.
    Splits preferentially on paragraph → newline → space boundaries.
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if end < len(text):
            split_at = -1
            for sep in separators:
                idx = chunk_text.rfind(sep)
                if idx > chunk_size // 2:
                    split_at = idx + len(sep)
                    break
            if split_at > 0:
                end = start + split_at
                chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "char_start": start, "char_end": end})
        if end >= len(text):  # reached the end — avoid tail micro-chunk loop
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


# %% Page-boundary lookup
def build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    """
    Build a list of (char_start, char_end, page_number) tuples from the
    per-page text list (joined with \\n\\n between pages).
    """
    page_map: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        end = pos + len(p["text"])
        page_map.append((pos, end, p["page"]))
        pos = end + 2  # account for the \n\n join separator
    return page_map


def char_range_to_pages(char_start: int, char_end: int, page_map: list) -> tuple[int, int]:
    """Return (first_page, last_page) that the char range [char_start, char_end) overlaps."""
    first = last = None
    for seg_start, seg_end, pg in page_map:
        if char_end <= seg_start:
            break
        if char_start < seg_end:
            if first is None:
                first = pg
            last = pg
    return (first or 1, last or 1)


def chunk_page_offsets(
    char_start: int,
    char_end: int,
    page_map: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """
    Return (page_char_start, page_char_end):
        page_char_start – char offset of chunk_start within the *first* page it touches.
        page_char_end   – char offset of chunk_end   within the *last*  page it touches.

    These are the offsets the PDF viewer (e.g. PDF.js text layer) needs so it
    can jump to the right position within each page without having to know about
    the full-document concatenation offset.
    """
    first_seg_start = last_seg_start = 0
    found_first = found_last = False
    for seg_start, seg_end, _pg in page_map:
        if char_end <= seg_start:
            break
        if char_start < seg_end:
            if not found_first:
                first_seg_start = seg_start
                found_first = True
            last_seg_start = seg_start
            found_last = True
    page_char_start = max(0, char_start - first_seg_start)
    page_char_end   = max(0, char_end   - last_seg_start)
    return page_char_start, page_char_end


# %% Constants
PDF_ROOT       = Path(__file__).resolve().parent.parent / "cuad_data" / "CUAD_v1" / "full_contract_pdf"
INDEX_NAME     = os.getenv("INDEX_NAME")
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "pdf_indexing_checkpoint.json"

MAX_PDFS         = int(os.getenv("MAX_PDFS", "0"))   # 0 = no limit
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
BULK_CHUNK_SIZE   = int(os.getenv("BULK_CHUNK_SIZE", "200"))

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


# %% Connect to OpenSearch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_search_connect import connect  # noqa: E402

client = connect()
client.info()


# %% Ensure index exists with the correct mapping
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
    },
    "mappings": {
        "properties": {
            # Chunk text — primary search field
            "text": {"type": "text", "analyzer": "english"},
            # Contract filename
            "title": {"type": "keyword"},
            # Character offsets within the full PDF text string
            "char_start": {"type": "integer"},
            "char_end":   {"type": "integer"},
            # PDF page range the chunk spans (PDF-only fields)
            "page_start":      {"type": "integer"},
            "page_end":        {"type": "integer"},
            # Char offset of chunk start/end *within its respective page* text.
            # Use these in the PDF viewer to jump to and highlight the exact text.
            "page_char_start": {"type": "integer"},
            "page_char_end":   {"type": "integer"},
            # Relative path to source PDF (for reference)
            "pdf_path":        {"type": "keyword"},
            # all-MiniLM-L6-v2 — 384-dim vectors
            "embedding": {
                "type":      "knn_vector",
                "dimension": 384,
                "method": {
                    "name":       "hnsw",
                    "space_type": "cosinesimil",
                    "engine":     "lucene",
                },
            },
        }
    },
}

if client.indices.exists(index=INDEX_NAME):
    print(f"Index '{INDEX_NAME}' already exists — skipping creation.")
else:
    resp = client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
    print(f"Index '{INDEX_NAME}' created: {resp}")


# Reduce memory pressure during bulk indexing
client.indices.put_settings(
    index=INDEX_NAME,
    body={"index": {"refresh_interval": "-1", "number_of_replicas": "0"}},
)


# %% Discover PDFs (case-insensitive .pdf / .PDF)
def find_pdfs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.upper() == ".PDF"
    )


all_pdfs = find_pdfs(PDF_ROOT)
if MAX_PDFS > 0:
    all_pdfs = all_pdfs[:MAX_PDFS]

print(f"PDFs discovered : {len(all_pdfs)}")
print(f"Index           : {INDEX_NAME}")
print(f"Chunk size      : {CHUNK_SIZE}  overlap : {CHUNK_OVERLAP}")


# %% Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Embedding model loaded.")


# %% Checkpoint helpers
def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"last_doc_id": None, "doc_count": 0}


def save_checkpoint(last_doc_id: str | None, doc_count: int) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"last_doc_id": last_doc_id, "doc_count": doc_count}))


checkpoint         = load_checkpoint()
last_indexed_doc_id = checkpoint["last_doc_id"]
starting_doc_count  = checkpoint["doc_count"]
print(f"Resuming from doc_id={last_indexed_doc_id}, count={starting_doc_count}")


# %% Chunk iterator — yields one flat dict per chunk across all PDFs
def iter_chunks():
    """
    For every PDF, extract text page-by-page, join into a single string,
    split into overlapping chunks, and yield a dict per chunk.

    Each dict:
        id              – "{title}-chunk-{n}"
        title           – PDF stem (filename without extension)
        text            – chunk text
        char_start      – start offset within the full concatenated PDF text
        char_end        – end   offset within the full concatenated PDF text
        page_start      – first PDF page the chunk overlaps
        page_end        – last  PDF page the chunk overlaps
        page_char_start – char offset of chunk start within page_start's text
        page_char_end   – char offset of chunk end   within page_end's text
        pdf_path        – relative path to the source PDF (for reference)
    """
    extraction_errors = 0
    for pdf_path in all_pdfs:
        title = pdf_path.stem  # filename without .PDF extension

        try:
            pages = extract_pages_from_pdf(pdf_path)
        except Exception as exc:
            print(f"\n[WARN] Could not extract {pdf_path.name}: {exc}")
            extraction_errors += 1
            continue

        # Filter out empty pages and clean whitespace
        pages = [p for p in pages if p["text"].strip()]
        if not pages:
            print(f"\n[WARN] No text extracted from {pdf_path.name} — skipping")
            continue

        full_text = "\n\n".join(p["text"] for p in pages)
        page_map  = build_page_map(pages)
        rel_path  = str(pdf_path.relative_to(PDF_ROOT))

        for chunk_idx, chunk in enumerate(split_text_with_offsets(full_text, CHUNK_SIZE, CHUNK_OVERLAP)):
            pg_start, pg_end = char_range_to_pages(
                chunk["char_start"], chunk["char_end"], page_map
            )
            pg_char_start, pg_char_end = chunk_page_offsets(
                chunk["char_start"], chunk["char_end"], page_map
            )
            yield {
                "id":              f"{title}-chunk-{chunk_idx}",
                "title":           title,
                "text":            chunk["text"],
                "char_start":      chunk["char_start"],
                "char_end":        chunk["char_end"],
                "page_start":      pg_start,
                "page_end":        pg_end,
                "page_char_start": pg_char_start,
                "page_char_end":   pg_char_end,
                "pdf_path":        rel_path,
            }

    if extraction_errors:
        print(f"\n[INFO] PDF extraction errors: {extraction_errors}")


# %% Bulk indexing generator — encode + yield OpenSearch actions
def index_docs_bulk():
    batch_docs:  list[dict] = []
    batch_texts: list[str]  = []
    skip_mode = last_indexed_doc_id is not None

    def _flush(docs, texts):
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=False,
        )
        for j, d in enumerate(docs):
            yield {
                "_index": INDEX_NAME,
                "_id":    d["id"],
                "_source": {
                    "title":           d["title"],
                    "text":            d["text"],
                    "char_start":      d["char_start"],
                    "char_end":        d["char_end"],
                    "page_start":      d["page_start"],
                    "page_end":        d["page_end"],
                    "page_char_start": d["page_char_start"],
                    "page_char_end":   d["page_char_end"],
                    "pdf_path":        d["pdf_path"],
                    "embedding":       embeddings[j].tolist(),
                },
            }

    for chunk in iter_chunks():
        # Resume: skip already-indexed chunks
        if skip_mode:
            if chunk["id"] == last_indexed_doc_id:
                skip_mode = False
            continue

        batch_docs.append(chunk)
        batch_texts.append(chunk["text"])

        if len(batch_docs) >= ENCODE_BATCH_SIZE:
            yield from _flush(batch_docs, batch_texts)
            batch_docs.clear()
            batch_texts.clear()

    if batch_docs:
        yield from _flush(batch_docs, batch_texts)


# %% Run bulk indexing
start_time    = time.time()
doc_count     = starting_doc_count
error_count   = 0
last_doc_id   = last_indexed_doc_id
batch_counter = 0

with tqdm(total=None, desc="Indexing PDF chunks", initial=starting_doc_count, unit="chunk") as pbar:
    for success, info in helpers.streaming_bulk(
        client,
        index_docs_bulk(),
        chunk_size=BULK_CHUNK_SIZE,
        max_chunk_bytes=5 * 1024 * 1024,
        request_timeout=120,
        raise_on_error=False,
    ):
        if success:
            doc_count += 1
            last_doc_id = info.get("index", {}).get("_id") or info.get("_id")
            batch_counter += 1
            if batch_counter % 10_000 == 0:
                save_checkpoint(last_doc_id, doc_count)
        else:
            error_count += 1
            print(f"\n[ERROR] Bulk error: {info}")
        pbar.update(1)

save_checkpoint(last_doc_id, doc_count)


# %% Finalise — restore index settings
client.indices.put_settings(
    index=INDEX_NAME,
    body={"index": {"refresh_interval": "1s", "number_of_replicas": "0"}},
)
client.indices.refresh(index=INDEX_NAME)


# %% Summary
elapsed       = time.time() - start_time
newly_indexed = doc_count - starting_doc_count
count_in_index = client.count(index=INDEX_NAME)["count"]

print("\n====== PDF INGESTION COMPLETE ======")
print(f"Extractor              : {EXTRACTOR_NAME}")
print(f"PDFs processed         : {len(all_pdfs)}")
print(f"Chunks indexed this run: {newly_indexed}")
print(f"Total chunks indexed   : {doc_count}")
print(f"Errors                 : {error_count}")
print(f"Elapsed time           : {elapsed:.2f} seconds")
print(f"Docs in index          : {count_in_index}")

client.transport.close()
