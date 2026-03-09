# %% Imports & docstring
"""
upload_to_qdrant.py
───────────────────
Extracts text from CUAD PDF contracts, generates embeddings with
all-MiniLM-L6-v2, and uploads up to 1 000 chunks to a Qdrant Cloud
collection named 'cuad_contracts'.

Reuses the PDF extraction and chunking logic from:
  project1/open_search_tuning/cuad_opensearch/notebooks/extract_index_cuad_pdfs.py

Usage
─────
    python upload_to_qdrant.py

Environment (loaded from .env in this directory)
─────────────────────────────────────────────────
    QDRANT_API_KEY   – Qdrant Cloud API key
    CLUSTER_URL      – Qdrant Cloud cluster URL (without port)

Optional env vars
─────────────────
    MAX_DOCS         – max chunks to upload (default: 1000)
    CHUNK_SIZE       – characters per chunk  (default: 500)
    CHUNK_OVERLAP    – overlap between chunks (default: 50)
    ENCODE_BATCH_SIZE– embedding batch size   (default: 32)
    UPLOAD_BATCH_SIZE– upsert batch size       (default: 100)
"""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

print("[INFO] Imports loaded successfully.")

# %% Configuration — load .env and set constants
load_dotenv(Path(__file__).resolve().parent / ".env")

QDRANT_API_KEY    = os.environ["QDRANT_API_KEY"]
CLUSTER_URL       = os.environ["CLUSTER_URL"].strip()

COLLECTION_NAME   = "cuad_contracts"
VECTOR_SIZE       = 384  # all-MiniLM-L6-v2

MAX_DOCS          = int(os.getenv("MAX_DOCS", "1000"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))

PDF_ROOT = Path(
    "/home/ridhi/projects/project1/open_search_tuning"
    "/cuad_opensearch/cuad_data/CUAD_v1/full_contract_pdf"
)

print(f"[INFO] Config loaded:")
print(f"       CLUSTER_URL      = {CLUSTER_URL}")
print(f"       COLLECTION_NAME  = {COLLECTION_NAME}")
print(f"       VECTOR_SIZE      = {VECTOR_SIZE}")
print(f"       MAX_DOCS         = {MAX_DOCS}")
print(f"       CHUNK_SIZE       = {CHUNK_SIZE}  |  CHUNK_OVERLAP = {CHUNK_OVERLAP}")
print(f"       ENCODE_BATCH_SIZE= {ENCODE_BATCH_SIZE}  |  UPLOAD_BATCH_SIZE = {UPLOAD_BATCH_SIZE}")
print(f"       PDF_ROOT exists  = {PDF_ROOT.exists()}")


# %% PDF extraction backend — auto-selects best available library
def _make_extractor():
    try:
        import fitz

        def _extract(path: Path):
            pages = []
            with fitz.open(str(path)) as doc:
                for i, page in enumerate(doc):
                    pages.append({"page": i + 1, "text": page.get_text("text") or ""})
            return pages

        print("[DEBUG] PDF backend selected: pymupdf (fitz)")
        return "pymupdf", _extract
    except ImportError:
        print("[DEBUG] pymupdf not available, trying pdfplumber …")

    try:
        import pdfplumber

        def _extract(path: Path):
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    pages.append({"page": i + 1, "text": page.extract_text() or ""})
            return pages

        print("[DEBUG] PDF backend selected: pdfplumber")
        return "pdfplumber", _extract
    except ImportError:
        print("[DEBUG] pdfplumber not available, trying pdfminer …")

    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer

        def _extract(path: Path):
            pages = []
            for i, layout in enumerate(extract_pages(str(path))):
                text = "".join(
                    el.get_text()
                    for el in layout
                    if isinstance(el, LTTextContainer)
                )
                pages.append({"page": i + 1, "text": text})
            return pages

        print("[DEBUG] PDF backend selected: pdfminer")
        return "pdfminer", _extract
    except ImportError:
        pass

    raise RuntimeError("No PDF library found. Install pymupdf, pdfplumber, or pdfminer.six.")


EXTRACTOR_NAME, extract_pages_from_pdf = _make_extractor()
print(f"[INFO] PDF extractor : {EXTRACTOR_NAME}")


# %% Text chunking helpers
def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
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
        if end >= len(text):
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


# ── Page-boundary helpers ────────────────────────────────────────────────────
def build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    page_map = []
    pos = 0
    for p in pages:
        end = pos + len(p["text"])
        page_map.append((pos, end, p["page"]))
        pos = end + 2
    return page_map


def char_range_to_pages(char_start: int, char_end: int, page_map: list) -> tuple[int, int]:
    first = last = None
    for seg_start, seg_end, pg in page_map:
        if char_end <= seg_start:
            break
        if char_start < seg_end:
            if first is None:
                first = pg
            last = pg
    return (first or 1, last or 1)


# %% PDF discovery & chunk iterator
def find_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.upper() == ".PDF")


def iter_chunks(all_pdfs: list[Path], limit: int):
    """Yield chunk dicts until *limit* chunks have been emitted."""
    count = 0
    skipped_pdfs = 0
    for pdf_path in all_pdfs:
        if count >= limit:
            return
        title = pdf_path.stem
        print(f"[DEBUG] Extracting: {pdf_path.name}", end="\r")
        try:
            pages = extract_pages_from_pdf(pdf_path)
        except Exception as exc:
            print(f"\n[WARN] Could not extract {pdf_path.name}: {exc}")
            skipped_pdfs += 1
            continue

        pages = [p for p in pages if p["text"].strip()]
        if not pages:
            print(f"[WARN] No text extracted from {pdf_path.name} — skipping")
            skipped_pdfs += 1
            continue

        full_text = "\n\n".join(p["text"] for p in pages)
        page_map  = build_page_map(pages)
        rel_path  = str(pdf_path.relative_to(PDF_ROOT))
        print(f"[DEBUG] {title}: {len(pages)} pages, {len(full_text):,} chars", end="\r")

        for chunk_idx, chunk in enumerate(
            split_text_with_offsets(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            if count >= limit:
                return
            pg_start, pg_end = char_range_to_pages(
                chunk["char_start"], chunk["char_end"], page_map
            )
            yield {
                "doc_id":     f"{title}-chunk-{chunk_idx}",
                "title":      title,
                "text":       chunk["text"],
                "char_start": chunk["char_start"],
                "char_end":   chunk["char_end"],
                "page_start": pg_start,
                "page_end":   pg_end,
                "pdf_path":   rel_path,
            }
            count += 1

    if skipped_pdfs:
        print(f"\n[INFO] PDFs skipped due to extraction errors: {skipped_pdfs}")


# %% Connect to Qdrant and create collection if needed
print(f"[INFO] Connecting to Qdrant: {CLUSTER_URL}")
qdrant = QdrantClient(
    url=f"{CLUSTER_URL}:6333",
    api_key=QDRANT_API_KEY,
    timeout=30,          # seconds — avoids silent hangs if cluster is unreachable
    prefer_grpc=False,
)
print("[INFO] Qdrant client created.")

existing = [c.name for c in qdrant.get_collections().collections]
print(f"[DEBUG] Existing collections: {existing}")
if COLLECTION_NAME in existing:
    print(f"[INFO] Collection '{COLLECTION_NAME}' already exists — skipping creation.")
else:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[INFO] Collection '{COLLECTION_NAME}' created (dim={VECTOR_SIZE}, distance=Cosine).")


# %% Load embedding model
# NOTE: first run downloads ~90 MB from HuggingFace — this can take a minute.
#       Subsequent runs load from the local cache and are fast.
print("[INFO] Loading embedding model all-MiniLM-L6-v2 (downloading if not cached) …")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print(f"[INFO] Model loaded. Max sequence length: {model.max_seq_length} tokens.")


# %% Discover PDFs
all_pdfs = find_pdfs(PDF_ROOT)
print(f"[INFO] PDFs discovered : {len(all_pdfs)}")
print(f"[INFO] Target chunks   : {MAX_DOCS}  |  chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}")
if not all_pdfs:
    raise FileNotFoundError(f"No PDFs found under {PDF_ROOT}. Check PDF_ROOT path.")

# %% Encode & upload in batches
chunk_buffer: list[dict] = []
uploaded = 0
errors   = 0

def flush_buffer(buf: list[dict]) -> int:
    """Encode and upsert a batch; returns number of points uploaded."""
    texts      = [d["text"] for d in buf]
    print(f"[DEBUG] Encoding batch of {len(texts)} chunks …", end="\r")
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    print(f"[DEBUG] Embeddings shape: {embeddings.shape}  — upserting …", end="\r")
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d["doc_id"])),
            vector=embeddings[i].tolist(),
            payload={
                "doc_id":     d["doc_id"],
                "title":      d["title"],
                "text":       d["text"],
                "char_start": d["char_start"],
                "char_end":   d["char_end"],
                "page_start": d["page_start"],
                "page_end":   d["page_end"],
                "pdf_path":   d["pdf_path"],
            },
        )
        for i, d in enumerate(buf)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


with tqdm(total=MAX_DOCS, desc="Uploading chunks", unit="chunk") as pbar:
    for chunk in iter_chunks(all_pdfs, MAX_DOCS):
        chunk_buffer.append(chunk)
        if len(chunk_buffer) >= UPLOAD_BATCH_SIZE:
            try:
                n = flush_buffer(chunk_buffer)
                uploaded += n
                pbar.update(n)
                print(f"[INFO] Batch upserted — total uploaded so far: {uploaded}", end="\r")
            except Exception as exc:
                print(f"\n[ERROR] Upsert failed: {exc}")
                errors += 1
            chunk_buffer.clear()

    # Flush remainder
    if chunk_buffer:
        print(f"\n[INFO] Flushing final batch of {len(chunk_buffer)} chunks …")
        try:
            n = flush_buffer(chunk_buffer)
            uploaded += n
            pbar.update(n)
        except Exception as exc:
            print(f"\n[ERROR] Final upsert failed: {exc}")
            errors += 1
        chunk_buffer.clear()


# %% Summary
info = qdrant.get_collection(COLLECTION_NAME)
print("\n====== UPLOAD COMPLETE ======")
print(f"[INFO] Collection      : {COLLECTION_NAME}")
print(f"[INFO] Chunks uploaded : {uploaded}")
print(f"[INFO] Errors          : {errors}")
print(f"[INFO] Vectors in coll.: {info.points_count}")
if errors:
    print(f"[WARN] {errors} batch(es) failed — check [ERROR] lines above.")

# %%
