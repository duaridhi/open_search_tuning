# %%
print('Start')

# %%
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

IR_DATASET_NAME="msmarco-passage/train/split200-train"
INDEX_NAME = "ir-dataset-train-v2"     
CHECKPOINT_FILE = "indexing_checkpoint.json"


# %%
import open_search_connect
from open_search_connect import connect
client = connect()


# %%
import ir_datasets
dataset = ir_datasets.load(IR_DATASET_NAME)
client.count(index=INDEX_NAME)


# %%

#Loggin in
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set, skipping Hugging Face login")

#Load the model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# %%
from opensearchpy import OpenSearch, helpers
import ir_datasets
import time
from opensearchpy.helpers import bulk
from tqdm import tqdm

# %%
# CHECKPOINT FUNCTIONS

def load_checkpoint():
    """Load the last checkpoint (last indexed doc_id and count)"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_doc_id": None, "doc_count": 0}

def save_checkpoint(last_doc_id, doc_count):
    """Save checkpoint to file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_doc_id": last_doc_id, "doc_count": doc_count}, f)

# Load checkpoint
checkpoint = load_checkpoint()
last_indexed_doc_id = checkpoint["last_doc_id"]
starting_doc_count = checkpoint["doc_count"]

print(f"Loaded checkpoint: last_doc_id={last_indexed_doc_id}, doc_count={starting_doc_count}")

# %%
MAX_DOCS = 8841825
BATCH_SIZE = 1000   # safe value for laptops


def index_docs_bulk():
    docs_iter = dataset.docs_iter()
    batch = []
    texts = []
    doc_ids = []
    doc_index = 0
    skip_mode = last_indexed_doc_id is not None
    
    for i, doc in enumerate(docs_iter):
        if i >= MAX_DOCS:
            break
        
        # Skip documents until we reach the checkpoint
        if skip_mode:
            if doc.doc_id == last_indexed_doc_id:
                skip_mode = False
                print(f"Checkpoint found. Resuming from doc: {doc.doc_id}")
            continue
        
        batch.append(doc)
        texts.append(doc.text)
        doc_ids.append(doc.doc_id)
        
        # Encode in batches of 64 for speed
        if len(batch) >= 64:
            t0 = time.time()
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            
            for j, doc in enumerate(batch):
                yield {
                    "_index": INDEX_NAME,
                    "_id": doc.doc_id,
                    "_source": {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "text_vector": embeddings[j].tolist()
                    }
                }
            batch = []
            texts = []
            doc_ids = []
    
    # Process remaining documents
    if batch:
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        
        for j, doc in enumerate(batch):
            yield {
                "_index": INDEX_NAME,
                "_id": doc.doc_id,
                "_source": {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "text_vector": embeddings[j].tolist()
                }
            }

# -----------------------------
# BULK INDEX WITH PROGRESS
# TIME-EFFICIENT CHECKPOINTING
# Start with the count from the checkpoint
start_time = time.time()
doc_count = starting_doc_count
error_count = 0
last_doc_id = last_indexed_doc_id
batch_counter = 0

with tqdm(total=MAX_DOCS, desc="Indexing documents", initial=starting_doc_count) as pbar:
    for success, info in helpers.streaming_bulk(
        client,
        index_docs_bulk(),
        chunk_size=BATCH_SIZE,
        request_timeout=120,
    ):
        if success:
            doc_count += 1
            last_doc_id = info.get("_id")
            batch_counter += 1
            
            # Save checkpoint every 10,000 documents
            if batch_counter % 10000 == 0:
                save_checkpoint(last_doc_id, doc_count)
                print(f"\nCheckpoint saved at doc_count={doc_count}, last_doc_id={last_doc_id}")
        else:
            error_count += 1
        pbar.update(1)

end_time = time.time()

# Save final checkpoint
save_checkpoint(last_doc_id, doc_count)
print(f"\nFinal checkpoint saved at doc_count={doc_count}, last_doc_id={last_doc_id}")

# FINALIZE
# Reset refresh interval for better query performance
client.indices.put_settings(
    index=INDEX_NAME,
    body={"index": {"refresh_interval": "1s"}}
)
client.indices.refresh(index=INDEX_NAME)

# STATS
elapsed = end_time - start_time
newly_indexed = doc_count - starting_doc_count
rate = newly_indexed / elapsed if elapsed > 0 else 0

count_in_index = client.count(index=INDEX_NAME)["count"]

print("\n====== INGESTION COMPLETE ======")
print(f"Documents indexed in this run: {newly_indexed}")
print(f"Total documents indexed: {doc_count}")
print(f"Errors: {error_count}")
print(f"Elapsed time: {elapsed:.2f} seconds")     
print(f"Indexing rate: {rate:.2f} docs/sec")
print(f"Docs in index: {count_in_index}")

client.transport.close()



