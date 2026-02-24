# %%
# Load the IR dataset library and fetch the MS MARCO passage dataset split
import ir_datasets

dataset = ir_datasets.load("msmarco-passage/train/split200-train")

# %%
import ir_datasets

# %%
# Print dataset statistics to understand the size and structure
print("Qrels:", dataset.qrels_count())
print("Docs:", dataset.docs_count())
print("Queries:", dataset.queries_count())

# %%
# Build a dictionary mapping query IDs to query text for fast lookup
queries = {}
for query in dataset.queries_iter():
    queries[query.query_id] = query.text

# %%
# Build a dictionary mapping document IDs to document text (limited to first 1000 docs for memory efficiency)
docs = {}
count = 0

for doc in dataset.docs_iter():
    docs[doc.doc_id] = doc.text
    count += 1
    if count == 1000:
        break

# %%
# Test: retrieve a sample document by ID
print(docs.get('4') )

# %%
# Build a consolidated records list combining queries, documents, and relevance judgments
count=0
records = []

for qrel in dataset.qrels_iter():
    count +=1
    record = {
        "query_id": qrel.query_id,
        "query_text": queries.get(qrel.query_id),
        "doc_id": qrel.doc_id,
        "doc_text": docs.get(qrel.doc_id),
        "relevance": qrel.relevance
    }
    records.append(record)
   

print(count)

# %%
# Helper function to retrieve all records for a given query ID
def get_by_query_id(records, qid):
    return [r for r in records if r["query_id"] == qid]

print(get_by_query_id(records, "737889"))

# %%
# Helper function to retrieve all records for a given document ID
def get_by_doc_id(records, doc_id):
    return [r for r in records if r["doc_id"] == doc_id]

print(get_by_doc_id(records, "49"))

# %%
# Test query: retrieve records for a specific query ID
def get_by_query_id(records, qid):
    return [r for r in records if r["query_id"] == qid]

print(get_by_query_id(records, "1103168"))

# %%



