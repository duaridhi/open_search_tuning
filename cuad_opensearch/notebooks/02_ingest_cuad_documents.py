# filepath: /cuad_opensearch/cuad_opensearch/notebooks/02_ingest_cuad_documents.py

# %% Imports & environment setup
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from opensearchpy import helpers
import time
from tqdm import tqdm

# NOTE: The HuggingFace SQuAD-format dataset (theatticusproject/cuad) has one row
# per Q&A pair, resulting in ~41 duplicate contexts per contract. It is used for
# evaluation in 04_evaluate_*.py. Ingestion uses the local CUAD_v1.json which has
# exactly one {title, context} entry per contract (510 unique contracts).

load_dotenv()

# %% Minimal recursive character splitter (no external dependencies)
def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """Split text into chunks of ~chunk_size chars with overlap.
    Returns a list of dicts with keys: text, char_start, char_end.
    Splits preferentially on paragraph -> newline -> space boundaries.
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
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
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks

# %% Constants & configuration
CUAD_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "CUAD_v1.json"
INDEX_NAME = "cuad_dataset"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "indexing_checkpoint.json"

MAX_DOCS = 2  # Max source rows to process from the dataset
ENCODE_BATCH_SIZE = 32
BULK_CHUNK_SIZE = 200

# LangChain text splitter settings
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# %% Connect to OpenSearch
from open_search_connect import connect
client = connect()
client.info()

# Reduce OpenSearch memory pressure during bulk indexing
client.indices.put_settings(
    index=INDEX_NAME,
    body={
        "index": {
            "refresh_interval": "-1",
            "number_of_replicas": "0"
        }
    }
)

# %% Load CUAD contracts from local JSON (one unique contract per entry)
with open(CUAD_JSON_PATH) as f:
    cuad_contracts = json.load(f)["data"]  # list of {title, paragraphs}
print(f"Loaded {len(cuad_contracts)} unique contracts from {CUAD_JSON_PATH.name}")

# %% Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Embedding model loaded.")
print(f"Text splitter ready (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

# %% Checkpoint functions
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open('r') as f:
            return json.load(f)
    return {"last_doc_id": None, "doc_count": 0}

def save_checkpoint(last_doc_id, doc_count):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open('w') as f:
        json.dump({"last_doc_id": last_doc_id, "doc_count": doc_count}, f)

checkpoint = load_checkpoint()
last_indexed_doc_id = checkpoint["last_doc_id"]
starting_doc_count = checkpoint["doc_count"]
print(f"Resuming from doc_id={last_indexed_doc_id}, count={starting_doc_count}")

# %% Bulk indexing generator
def iter_chunks():
    """Yield flat chunk dicts for every row in the CUAD train split.

    Each dict contains:
        id         – unique chunk id: "{row_id}-chunk-{n}"
        title      – source contract filename
        text       – chunk text only
        char_start – start offset of this chunk within the original context
        char_end   – end offset of this chunk within the original context
    """
    for i, doc in enumerate(cuad_contracts):
        if i >= MAX_DOCS:
            break
        title = doc["title"]
        # Each CUAD entry has exactly one paragraph containing the full contract text
        context = doc["paragraphs"][0]["context"]
        for chunk_idx, chunk in enumerate(split_text_with_offsets(context, CHUNK_SIZE, CHUNK_OVERLAP)):
            yield {
                "id": f"{title}-chunk-{chunk_idx}",
                "title": title,
                "text": chunk["text"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
            }


def index_docs_bulk():
    """Batch-encode chunks and yield OpenSearch bulk actions."""
    batch_docs: list = []
    batch_texts: list = []
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
                "_id": d["id"],
                "_source": {
                    "title": d["title"],
                    "text": d["text"],
                    "char_start": d["char_start"],
                    "char_end": d["char_end"],
                    "embedding": embeddings[j].tolist(),
                },
            }

    for chunk in iter_chunks():
        # Resume support: skip already-indexed chunks
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

    # Flush remaining chunks
    if batch_docs:
        yield from _flush(batch_docs, batch_texts)

# %% Run bulk indexing
start_time = time.time()
doc_count = starting_doc_count
error_count = 0
last_doc_id = last_indexed_doc_id
batch_counter = 0

# Total chunks is unknown upfront; pass None for an unbounded progress bar
with tqdm(total=None, desc="Indexing CUAD chunks", initial=starting_doc_count, unit="chunk") as pbar:
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

            if batch_counter % 10000 == 0:
                save_checkpoint(last_doc_id, doc_count)
        else:
            error_count += 1

        pbar.update(1)

save_checkpoint(last_doc_id, doc_count)

# %% Finalize — restore index settings
client.indices.put_settings(
    index=INDEX_NAME,
    body={"index": {"refresh_interval": "1s", "number_of_replicas": "0"}}
)
client.indices.refresh(index=INDEX_NAME)

# %% Stats & summary
elapsed = time.time() - start_time
newly_indexed = doc_count - starting_doc_count
count_in_index = client.count(index=INDEX_NAME)["count"]

print("\n====== INGESTION COMPLETE ======")
print(f"Documents indexed in this run: {newly_indexed}")
print(f"Total documents indexed:       {doc_count}")
print(f"Errors:                        {error_count}")
print(f"Elapsed time:                  {elapsed:.2f} seconds")
print(f"Docs in index:                 {count_in_index}")

client.transport.close()
# %%


client.indices.put_settings(
    index=INDEX_NAME,
    body={
        "index": {
            "refresh_interval": "1s",   # default
            "number_of_replicas": "1"   # default
        }
    }
)
