# %% Imports & docstring
"""
upload_to_qdrant_hf.py
───────────────────────
Same as upload_to_qdrant.py but uses the HuggingFace Inference API for
embeddings instead of a local sentence-transformers model.  No local GPU/RAM
cost for the embedding step — embeddings are generated server-side.

Advantages over upload_to_qdrant.py:
  - Large models (gte-Qwen2-1.5B-instruct, E5-mistral-7B-instruct, …) work
    without downloading or fitting in local RAM.
  - No CUDA/PyTorch dependency for ingest.

Limitations:
  - Requires HF_TOKEN with Inference API access.
  - Rate-limited by the HuggingFace free tier; reduce ENCODE_BATCH_SIZE (to
    8 or 4) and add ENCODE_SLEEP_S if you get 429 errors.
  - Only models whose feature-extraction endpoint returns pooled sentence
    embeddings produce correct results.  Models that return token-level
    embeddings need explicit mean-pooling and are not suitable here without
    modification.  Safe families: sentence-transformers/*, Alibaba-NLP/gte-*,
    BAAI/bge-*, intfloat/e5-*, nomic-ai/nomic-embed-text-*.

Text source: PDFs (same as upload_to_qdrant.py — PDF char offsets are
required by the UI for highlight rendering).

Usage
─────
    python cuad-demo-quadrant/upload_to_qdrant_hf.py

Environment (loaded from .env in cuad-demo-quadrant/)
──────────────────────────────────────────────────────
    HF_TOKEN        – required; HuggingFace Inference API token
    QDRANT_URL      – Local Qdrant URL (default: http://localhost:6333)
    CLUSTER_URL     – Qdrant Cloud cluster URL (takes precedence)
    QDRANT_API_KEY  – Cloud API key
    PDF_ROOT        – override path to full_contract_pdf/ directory

Optional env vars
─────────────────
    EMBED_MODEL       – HF model repo ID  (default: sentence-transformers/all-MiniLM-L6-v2)
    VECTOR_SIZE       – embedding dim; auto-detected via probe call if not set
    DOC_OFFSET        – skip first N PDFs alphabetically (default: 0)
    DOC_COUNT         – max PDFs to process, 0 = no limit (default: 0)
    MAX_DOCS          – max chunks to upload (default: 500000)
    CHUNK_SIZE        – characters per chunk (default: 500)
    CHUNK_OVERLAP     – overlap between chunks (default: 50)
    ENCODE_BATCH_SIZE – texts per API call (default: 32; lower to 8 if rate-limited)
    ENCODE_SLEEP_S    – seconds to sleep between API calls (default: 0)
    UPLOAD_BATCH_SIZE – Qdrant upsert batch size (default: 100)
    SKIP_INGESTED_DOCS– skip titles already in Qdrant (default: 1)
"""

import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import requests
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, SparseVector, SparseVectorParams, VectorParams
from tqdm import tqdm

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

COLLECTION_NAME    = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOC_OFFSET         = int(os.getenv("DOC_OFFSET", "0"))
DOC_COUNT          = int(os.getenv("DOC_COUNT", "0"))   # 0 = no doc-count limit

MAX_DOCS           = int(os.getenv("MAX_DOCS", "500000"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "50"))
ENCODE_BATCH_SIZE  = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
ENCODE_SLEEP_S     = float(os.getenv("ENCODE_SLEEP_S", "0"))
UPLOAD_BATCH_SIZE  = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))
SKIP_INGESTED_DOCS = os.getenv("SKIP_INGESTED_DOCS", "1").lower() not in ("0", "false", "no")
ENABLE_HYBRID    = os.getenv("ENABLE_HYBRID", "0").lower() not in ("0", "false", "no")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")

_CUAD_DATA_BASE = Path(
    "/home/ridhi/projects/project1/open_search_tuning"
    "/cuad_opensearch/cuad_data/CUAD_v1"
)
PDF_ROOT = Path(os.getenv("PDF_ROOT", str(_CUAD_DATA_BASE / "full_contract_pdf")))

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is required for the HF Inference API ingest path.")

logger.info(
    "Config loaded: COLLECTION=%s  EMBED_MODEL=%s  DOC_OFFSET=%d  DOC_COUNT=%d  MAX_DOCS=%d  "
    "CHUNK_SIZE=%d  CHUNK_OVERLAP=%d  ENCODE_BATCH=%d  SLEEP=%ss  UPLOAD_BATCH=%d  PDF_ROOT_EXISTS=%s",
    COLLECTION_NAME, EMBED_MODEL, DOC_OFFSET, DOC_COUNT, MAX_DOCS,
    CHUNK_SIZE, CHUNK_OVERLAP, ENCODE_BATCH_SIZE, ENCODE_SLEEP_S, UPLOAD_BATCH_SIZE, PDF_ROOT.exists(),
)

# %% HF Inference API — direct HTTP, no provider routing layer
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")
_HF_EMBED_URL = f"https://router.huggingface.co/{HF_PROVIDER}/models/{EMBED_MODEL}/pipeline/feature-extraction"
_HF_EMBED_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


_HF_EMBED_URL_V1 = f"https://router.huggingface.co/{HF_PROVIDER}/v1/embeddings"


def _hf_api_embed(texts: list[str]) -> list:
    # Try pipeline/feature-extraction first; fall back to OpenAI-compatible
    # /v1/embeddings (used by Scaleway and other TEI-backed providers).
    resp = requests.post(
        _HF_EMBED_URL,
        headers=_HF_EMBED_HEADERS,
        json={"inputs": texts},
        timeout=60,
    )
    if resp.status_code == 400:
        logger.debug("feature-extraction 400 body: %s", resp.text[:300])
        resp2 = requests.post(
            _HF_EMBED_URL_V1,
            headers=_HF_EMBED_HEADERS,
            json={"model": EMBED_MODEL, "input": texts},
            timeout=60,
        )
        if resp2.ok:
            data = resp2.json()
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        logger.debug("/v1/embeddings %d body: %s", resp2.status_code, resp2.text[:300])
    resp.raise_for_status()
    return resp.json()


# %% Embedding helper — batched API calls with retry + optional sleep

def _pool(raw) -> np.ndarray:
    """Convert raw feature_extraction output to a (n_texts, dim) float32 array.

    HF models return one of three shapes:
      - (n_texts, dim)         — pooled sentence embeddings, use as-is
      - (n_texts, n_tokens, dim) — token-level, mean-pool over tokens
      - ragged list of (n_tokens_i, dim) — token-level with variable lengths

    Mean-pooling over tokens is the standard approach for converting a
    token-level model into sentence embeddings.
    """
    if isinstance(raw, np.ndarray):
        if raw.ndim == 2:
            return raw.astype(np.float32)
        if raw.ndim == 3:
            return raw.mean(axis=1).astype(np.float32)

    # raw is a list; items are either 1-D (pooled) or 2-D (token-level)
    pooled = []
    for item in raw:
        arr = np.array(item, dtype=np.float32)
        if arr.ndim == 2:       # token-level: (n_tokens, dim) → mean over tokens
            arr = arr.mean(axis=0)
        pooled.append(arr)
    return np.stack(pooled)


def _encode(texts: list[str]) -> np.ndarray:
    """Call HF feature_extraction in sub-batches; return L2-normalised float32 array."""
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = texts[i : i + ENCODE_BATCH_SIZE]
        for attempt in range(3):
            try:
                raw = _hf_api_embed(batch)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 5 * (2 ** attempt)   # 5 s, 10 s
                logger.warning(
                    "feature_extraction failed (attempt %d/3): %s — retrying in %ds",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        all_embs.append(_pool(raw))
        if ENCODE_SLEEP_S > 0:
            time.sleep(ENCODE_SLEEP_S)

    result = np.vstack(all_embs)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.where(norms == 0, 1.0, norms)
    return result


# %% VECTOR_SIZE detection (needs _pool to be defined first)
_vector_size_env = os.getenv("VECTOR_SIZE", "")
if _vector_size_env:
    VECTOR_SIZE = int(_vector_size_env)
    logger.info("VECTOR_SIZE from env: %d", VECTOR_SIZE)
else:
    logger.info("VECTOR_SIZE not set — probing model with a test embedding ...")
    _probe = _pool(_hf_api_embed(["probe"]))
    VECTOR_SIZE = int(_probe.shape[-1])
    logger.info("Auto-detected VECTOR_SIZE: %d  (output shape: %s)", VECTOR_SIZE, _probe.shape)


# %% Sparse model (hybrid search) — loaded only when ENABLE_HYBRID=1
_sparse_encoder = None
if ENABLE_HYBRID:
    try:
        from fastembed import SparseTextEmbedding
        logger.info("Loading sparse embedding model: %s", SPARSE_MODEL_NAME)
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        logger.info("Sparse model loaded.")
    except ImportError:
        logger.warning("fastembed not installed; ENABLE_HYBRID ignored. pip install fastembed")
        ENABLE_HYBRID = False


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
                    el.get_text() for el in layout if isinstance(el, LTTextContainer)
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


# %% Page-boundary helpers
def build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    """Return [(seg_start, seg_end, page_number), ...] for every page."""
    page_map = []
    pos = 0
    for p in pages:
        end = pos + len(p["text"])
        page_map.append((pos, end, p["page"]))
        pos = end + 2  # accounts for the "\n\n" separator
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


def iter_chunks(all_pdfs: list[Path], limit: int, skip_titles: set[str] | None = None):
    """Yield chunk dicts until *limit* chunks have been emitted."""
    count = 0
    skipped_pdfs = 0
    resumed_pdfs = 0
    for pdf_path in all_pdfs:
        if count >= limit:
            return
        title = pdf_path.stem
        if skip_titles and title in skip_titles:
            resumed_pdfs += 1
            continue
        try:
            pages = extract_pages_from_pdf(pdf_path)
        except Exception as exc:
            logger.warning("Could not extract %s: %s", pdf_path.name, exc)
            skipped_pdfs += 1
            continue

        for p in pages:
            if not p["text"].strip():
                logger.warning(
                    "Empty page %d in %s — keeping for offset continuity",
                    p["page"], pdf_path.name,
                )

        if not any(p["text"].strip() for p in pages):
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
            yield {
                "doc_id":            f"{title}-chunk-{chunk_idx}",
                "title":             title,
                "text":              chunk["text"],
                "char_start":        chunk["char_start"],
                "char_end":          chunk["char_end"],
                "page_start":        pg_start,
                "page_end":          pg_end,
                "page_offset_start": char_pos_to_page_offset(chunk["char_start"], pg_start, page_map),
                "page_offset_end":   char_pos_to_page_offset(chunk["char_end"],   pg_end,   page_map),
                "pdf_path":          rel_path,
            }
            count += 1

    if resumed_pdfs:
        logger.info("PDFs skipped (already ingested): %d", resumed_pdfs)
    if skipped_pdfs:
        logger.warning("PDFs skipped due to extraction errors: %d", skipped_pdfs)


# %% Connect to Qdrant and create collection if needed
qdrant = get_qdrant_client()
logger.info("Qdrant client initialized.")

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME in existing:
    logger.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)
else:
    create_kwargs: dict = dict(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    if ENABLE_HYBRID:
        create_kwargs["sparse_vectors_config"] = {"sparse": SparseVectorParams()}
        logger.info("ENABLE_HYBRID=1: adding sparse vector field to collection schema.")
    qdrant.create_collection(**create_kwargs)
    logger.info("Collection '%s' created (dim=%d, distance=Cosine, hybrid=%s).", COLLECTION_NAME, VECTOR_SIZE, ENABLE_HYBRID)

try:
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Keyword index ensured on %s.title", COLLECTION_NAME)
except Exception as exc:
    logger.warning("Could not create keyword index on %s.title: %s", COLLECTION_NAME, exc)


# %% Build resume set — titles already fully indexed
ingested_titles: set[str] = set()
if SKIP_INGESTED_DOCS and COLLECTION_NAME in existing:
    logger.info("SKIP_INGESTED_DOCS=1: scanning collection for already-ingested titles …")
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["title"],
            with_vectors=False,
        )
        for pt in results:
            ingested_titles.add(pt.payload.get("title", ""))
        if offset is None:
            break
    logger.info("RESUME: %d titles already in collection — will skip them.", len(ingested_titles))
else:
    logger.info("SKIP_INGESTED_DOCS=0 or fresh collection — processing all PDFs.")


# %% Discover PDFs
if not PDF_ROOT.exists():
    raise FileNotFoundError(
        f"PDF_ROOT does not exist: {PDF_ROOT}\n"
        "Set the PDF_ROOT env var or run cuad_download_utils.download_cuad_dataset() first."
    )
all_pdfs = find_pdfs(PDF_ROOT)
if DOC_COUNT > 0:
    all_pdfs = all_pdfs[DOC_OFFSET : DOC_OFFSET + DOC_COUNT]
elif DOC_OFFSET > 0:
    all_pdfs = all_pdfs[DOC_OFFSET:]
logger.info(
    "PDFs selected: %d (offset=%d, count=%d)  |  target chunks: %d  chunk_size=%d  overlap=%d",
    len(all_pdfs), DOC_OFFSET, DOC_COUNT, MAX_DOCS, CHUNK_SIZE, CHUNK_OVERLAP,
)
if not all_pdfs:
    raise FileNotFoundError(f"No PDFs found under {PDF_ROOT}. Check PDF_ROOT path.")


# %% Encode & upload in batches
chunk_buffer: list[dict] = []
uploaded = 0
errors   = 0


def flush_buffer(buf: list[dict]) -> int:
    """Encode via HF API and upsert a batch; returns number of points uploaded."""
    texts = [d["text"] for d in buf]
    embeddings = _encode(texts)

    sparse_vecs = None
    if ENABLE_HYBRID and _sparse_encoder is not None:
        sparse_vecs = list(_sparse_encoder.embed(texts))

    points = []
    for i, d in enumerate(buf):
        if sparse_vecs is not None:
            sv = sparse_vecs[i]
            vector = {
                "": embeddings[i].tolist(),
                "sparse": SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
            }
        else:
            vector = embeddings[i].tolist()
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d["doc_id"])),
            vector=vector,
            payload={
                "doc_id":            d["doc_id"],
                "title":             d["title"],
                "text":              d["text"],
                "char_start":        d["char_start"],
                "char_end":          d["char_end"],
                "page_start":        d["page_start"],
                "page_end":          d["page_end"],
                "page_offset_start": d["page_offset_start"],
                "page_offset_end":   d["page_offset_end"],
                "pdf_path":          d["pdf_path"],
            },
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


with tqdm(total=MAX_DOCS, desc="Uploading chunks", unit="chunk") as pbar:
    for chunk in iter_chunks(all_pdfs, MAX_DOCS, skip_titles=ingested_titles or None):
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
