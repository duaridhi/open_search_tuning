# filepath: /cuad_opensearch/cuad_opensearch/notebooks/02_ingest_cuad_documents.py

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from open_search_connect import connect
import ir_datasets
from sentence_transformers import SentenceTransformer
from opensearchpy import helpers
import time
from tqdm import tqdm

load_dotenv()

# Define constants
CUAD_DATASET_NAME = "theatticusproject/cuad-qa"
INDEX_NAME = "cuad-dataset"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "indexing_checkpoint.json"

# Connect to OpenSearch
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

# Load CUAD dataset
from datasets import load_dataset

# Load the CUAD QA dataset
dataset = load_dataset(CUAD_DATASET_NAME)
client.count(index=INDEX_NAME)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Checkpoint functions
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

MAX_DOCS = 15000  # Adjust based on CUAD dataset size
ENCODE_BATCH_SIZE = 32
BULK_CHUNK_SIZE = 200

def index_docs_bulk():
    docs_iter = dataset.docs_iter()
    batch = []
    texts = []
    skip_mode = last_indexed_doc_id is not None

    for i, doc in enumerate(docs_iter):
        if i >= MAX_DOCS:
            break

        if skip_mode:
            if doc.doc_id == last_indexed_doc_id:
                skip_mode = False
            continue

        batch.append(doc)
        texts.append(doc.text)

        if len(batch) >= ENCODE_BATCH_SIZE:
            embeddings = embedding_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=False,
            )

            for j, d in enumerate(batch):
                yield {
                    "_index": INDEX_NAME,
                    "_id": d.doc_id,
                    "_source": {
                        "doc_id": d.doc_id,
                        "text": d.text,
                        "text_vector": embeddings[j].tolist()
                    }
                }
            batch.clear()
            texts.clear()

    if batch:
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=False,
        )
        for j, d in enumerate(batch):
            yield {
                "_index": INDEX_NAME,
                "_id": d.doc_id,
                "_source": {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "text_vector": embeddings[j].tolist()
                }
            }

start_time = time.time()
doc_count = starting_doc_count
error_count = 0
last_doc_id = last_indexed_doc_id
batch_counter = 0

with tqdm(total=MAX_DOCS, desc="Indexing CUAD documents", initial=starting_doc_count) as pbar:
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

# FINALIZE — restore settings
client.indices.put_settings(
    index=INDEX_NAME,
    body={"index": {"refresh_interval": "1s", "number_of_replicas": "0"}}
)
client.indices.refresh(index=INDEX_NAME)

# STATS
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