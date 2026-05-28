# %% Imports & docstring
"""
upload_to_qdrant.py
───────────────────
Extracts text from CUAD PDF contracts, generates embeddings with
all-MiniLM-L6-v2, and uploads chunks to a Qdrant collection named 'cuad_contracts'.

Reuses the PDF extraction and chunking logic from:
  project1/open_search_tuning/cuad_opensearch/notebooks/extract_index_cuad_pdfs.py

Usage
─────
    python upload_to_qdrant.py

Environment (loaded from .env in this directory)
─────────────────────────────────────────────────
    QDRANT_URL      – Local Qdrant URL (default: http://localhost:6333)
    CLUSTER_URL     – Qdrant Cloud cluster URL (takes precedence over QDRANT_URL)
    QDRANT_API_KEY  – Cloud API key

Optional env vars
─────────────────
    MAX_DOCS         – max chunks to upload (default: 1000)
    CHUNK_SIZE       – characters per chunk  (default: 500)
    CHUNK_OVERLAP    – overlap between chunks (default: 50)
    ENCODE_BATCH_SIZE– embedding batch size   (default: 32)
    UPLOAD_BATCH_SIZE– upsert batch size       (default: 100)
"""

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from tqdm import tqdm

# Import cluster connection
from qdrant_cluster_connect import get_qdrant_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Imports loaded successfully.")

# %% Configuration — load .env and set constants
load_dotenv(Path(__file__).resolve().parent / ".env")

COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
VECTOR_SIZE       = 384  # all-MiniLM-L6-v2

MAX_DOCS          = int(os.getenv("MAX_DOCS", "500000"))  # effectively uncapped for full CUAD corpus
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))

PDF_ROOT = Path(
    "/home/ridhi/projects/project1/open_search_tuning"
    "/cuad_opensearch/cuad_data/CUAD_v1/full_contract_pdf"
)

logger.info(
    "Config loaded: COLLECTION=%s  VECTOR_SIZE=%d  MAX_DOCS=%d  "
    "CHUNK_SIZE=%d  CHUNK_OVERLAP=%d  ENCODE_BATCH=%d  UPLOAD_BATCH=%d  PDF_ROOT_EXISTS=%s",
    COLLECTION_NAME, VECTOR_SIZE, MAX_DOCS,
    CHUNK_SIZE, CHUNK_OVERLAP, ENCODE_BATCH_SIZE, UPLOAD_BATCH_SIZE, PDF_ROOT.exists(),
)


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

        logger.info("PDF backend selected: pymupdf (fitz)")
        return "pymupdf", _extract
    except ImportError:
        logger.debug("pymupdf not available, trying pdfplumber ...")

    try:
        import pdfplumber

        def _extract(path: Path):
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    pages.append({"page": i + 1, "text": page.extract_text() or ""})
            return pages

        logger.info("PDF backend selected: pdfplumber")
        return "pdfplumber", _extract
    except ImportError:
        logger.debug("pdfplumber not available, trying pdfminer ...")

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

        logger.info("PDF backend selected: pdfminer")
        return "pdfminer", _extract
    except ImportError:
        pass

    raise RuntimeError("No PDF library found. Install pymupdf, pdfplumber, or pdfminer.six.")


EXTRACTOR_NAME, extract_pages_from_pdf = _make_extractor()
logger.info("PDF extractor: %s", EXTRACTOR_NAME)


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


def char_pos_to_page_offset(char_pos: int, page_number: int, page_map: list) -> int:
    for seg_start, seg_end, pg in page_map:
        if pg == page_number:
            return max(0, min(char_pos, seg_end) - seg_start)
    return 0


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
        try:
            pages = extract_pages_from_pdf(pdf_path)
        except Exception as exc:
            logger.warning("Could not extract %s: %s", pdf_path.name, exc)
            skipped_pdfs += 1
            continue

        pages = [p for p in pages if p["text"].strip()]
        if not pages:
            logger.warning("No text extracted from %s — skipping", pdf_path.name)
            skipped_pdfs += 1
            continue

        full_text = "\n\n".join(p["text"] for p in pages)
        page_map  = build_page_map(pages)
        rel_path  = str(pdf_path.relative_to(PDF_ROOT))

        for chunk_idx, chunk in enumerate(
            split_text_with_offsets(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            if count >= limit:
                return
            pg_start, pg_end = char_range_to_pages(
                chunk["char_start"], chunk["char_end"], page_map
            )
            page_offset_start = char_pos_to_page_offset(
                chunk["char_start"], pg_start, page_map
            )
            page_offset_end = char_pos_to_page_offset(
                chunk["char_end"], pg_end, page_map
            )
            yield {
                "doc_id":     f"{title}-chunk-{chunk_idx}",
                "title":      title,
                "text":       chunk["text"],
                "char_start": chunk["char_start"],
                "char_end":   chunk["char_end"],
                "page_start": pg_start,
                "page_end":   pg_end,
                "page_offset_start": page_offset_start,
                "page_offset_end":   page_offset_end,
                "pdf_path":   rel_path,
            }
            count += 1

    if skipped_pdfs:
        logger.warning("PDFs skipped due to extraction errors: %d", skipped_pdfs)


# %% Connect to Qdrant and create collection if needed
qdrant = get_qdrant_client()
logger.info("Qdrant client initialized.")

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME in existing:
    logger.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)
else:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Collection '%s' created (dim=%d, distance=Cosine).", COLLECTION_NAME, VECTOR_SIZE)

try:
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Keyword index ensured on %s.title", COLLECTION_NAME)
except Exception as exc:
    logger.warning("Could not create keyword index on %s.title: %s", COLLECTION_NAME, exc)


# %% Load embedding model
# NOTE: first run downloads ~90 MB from HuggingFace — this can take a minute.
#       Subsequent runs load from the local cache and are fast.
logger.info("Loading embedding model all-MiniLM-L6-v2 (downloading if not cached) ...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
logger.info("Model loaded. Max sequence length: %d tokens.", model.max_seq_length)


# %% Discover PDFs
all_pdfs = find_pdfs(PDF_ROOT)
logger.info("PDFs discovered: %d  |  target chunks: %d  chunk_size=%d  overlap=%d",
            len(all_pdfs), MAX_DOCS, CHUNK_SIZE, CHUNK_OVERLAP)
if not all_pdfs:
    raise FileNotFoundError(f"No PDFs found under {PDF_ROOT}. Check PDF_ROOT path.")

# %% Encode & upload in batches
chunk_buffer: list[dict] = []
uploaded = 0
errors   = 0

def flush_buffer(buf: list[dict]) -> int:
    """Encode and upsert a batch; returns number of points uploaded."""
    texts      = [d["text"] for d in buf]
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
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
                "page_offset_start": d["page_offset_start"],
                "page_offset_end":   d["page_offset_end"],
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
            except Exception as exc:
                logger.error("Upsert failed: %s", exc)
                errors += 1
            chunk_buffer.clear()

    # Flush remainder
    if chunk_buffer:
        logger.info("Flushing final batch of %d chunks ...", len(chunk_buffer))
        try:
            n = flush_buffer(chunk_buffer)
            uploaded += n
            pbar.update(n)
        except Exception as exc:
            logger.error("Final upsert failed: %s", exc)
            errors += 1
        chunk_buffer.clear()


# %% Summary
info = qdrant.get_collection(COLLECTION_NAME)
logger.info(
    "Upload complete — collection: %s  chunks_uploaded: %d  errors: %d  vectors_in_collection: %s",
    COLLECTION_NAME, uploaded, errors, info.points_count,
)
if errors:
    logger.warning("%d batch(es) failed — check error logs above.", errors)

# %%
