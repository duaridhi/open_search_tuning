# %%
print("Start")

# %%
from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ["HF_TOKEN"]
# %%
import open_search_connect 
client=open_search_connect.connect()
INDEX_NAME = "ir-dataset-train-v2"

# %%
# Set upthe transformer with no parallelism towork on wsl

#Parellelism=1
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

#Loggin in
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

#Load the model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
#Run a hybrid vector search on opensearch
def hybrid_opensearch_reranked(query, k):
    query_vector = embedding_model.encode(query).tolist()
    body ={
        "size": k,
        "query": {
            "hybrid": {
            "queries": [
                {
                "match": {
                    "text_field": {
                    "query": query
                    }
                }
                },
                {
                "knn": {
                    "text_vector": {
                     "k":10,   
                    "vector": query_vector,
                    "boost": 4.5

                    }
                }
                }
            ]
            }
        }
    }
    INDEX_NAME = "ir-dataset-train-v2"
    res = client.search(index=INDEX_NAME, body=body)
    hits = [(hit["_source"]["doc_id"], hit["_source"]["text"]) for hit in res["hits"]["hits"]]
    return hits


# %%
def rerank_with_cross_encoder(query, docs,rerank_count):
    pairs = [(query, text) for _, text in docs[:rerank_count]]
    scores = model.predict(pairs)

    reranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [(doc_id, text, score) for ((doc_id, text), score) in reranked]


# %%
#Try running a single query
query = "communication amid scientific minds"
single_docs = hybrid_opensearch_reranked(query,10)
print(single_docs)

# %%

# %%
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# %%
#Run the cross encoder
reranked_docs = rerank_with_cross_encoder(query,single_docs,10)
print(reranked_docs)

# %%
