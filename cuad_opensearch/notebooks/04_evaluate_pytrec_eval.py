"""
04_evaluate_pytrec_eval.py
Evaluate hybrid search on CUAD using pytrec_eval.

Ground-truth (qrels) is derived directly from the CUAD dataset:
  - query_id  : stable slug of the clause-type question (41 unique questions)
  - doc_id    : CUAD sample id  (unique per context)
  - relevance : 1 if answers["text"] is non-empty, else omitted from qrels
                (pytrec_eval only needs *relevant* docs listed)

Run dict is built by running hybrid search for every unique question.
"""
# %%
print("Starting evaluation")
# %%

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import pytrec_eval
from sentence_transformers import SentenceTransformer
# %%

# ── path setup so open_search_connect is importable ──────────────────────────
root_path = Path("/home/ridhi/projects/project1/open_search_tuning")
# s3_utils_path = root_path/"sr_utils"
# if str(s3_utils_path) not in sys.path:
#     sys.path.append(0, str(s3_utils_path))
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from cuad_opensearch.notebooks.open_search_connect import connect
from cuad_dataset_utils import (
    load_cuad_local_json,
    build_qrels,
    build_queries,
    _get_cuad_json_path,
)

load_dotenv(root_path / "notebooks/.env")

# %%
# ── constants ─────────────────────────────────────────────────────────────────
INDEX_NAME    = "cuad_dataset"
EMBEDDING_DIM = 384        # all-MiniLM-L6-v2
TOP_K         = 20         # docs retrieved per query
METRICS       = {          # pytrec_eval metric strings
    "map",
    "ndcg_cut_10",
    "P_10",
    "recall_10",
}


# ── hybrid search ─────────────────────────────────────────────────────────────

def _knn_query(vector: list[float], top_k: int) -> dict:
    return {
        "knn": {
            "embedding": {
                "vector": vector,
                "k": top_k,
            }
        }
    }


def _bm25_query(text: str) -> dict:
    return {
        "match": {
            "text": {
                "query": text,
                "analyzer": "english",
            }
        }
    }


def hybrid_search_run(
    client,
    embedding_model: SentenceTransformer,
    qid_to_question: dict[str, str],
    qrels: dict[str, dict[str, int]] | None = None,
    top_k: int = TOP_K,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run hybrid search for every query and return a pytrec_eval run dict.

    Fusion strategy: Reciprocal Rank Fusion (RRF, k=60) — parameter-free,
    robust default for combining BM25 + dense rankings.
    
    Parameters
    ----------
    client : OpenSearch client
    embedding_model : SentenceTransformer
    qid_to_question : dict[str, str]
        Mapping from query ID to question text
    qrels : dict[str, dict[str, int]], optional
        Ground truth relevance judgments for debugging output
    top_k : int
        Number of results to return per query
    verbose : bool
        If True, print debugging info for each query

    Returns
    -------
    run : {qid: {doc_id: rrf_score}}
    """
    run: dict[str, dict[str, float]] = {}
    query_count = 0

    for qid, question in tqdm(qid_to_question.items(), desc="Searching"):
        query_count += 1
        vector = embedding_model.encode(question, show_progress_bar=False).tolist()

        # ── BM25 ──────────────────────────────────────────────────────────────
        bm25_resp = client.search(
            index=INDEX_NAME,
            body={"query": _bm25_query(question), "size": top_k, "_source": False},
        )
        bm25_hits = [(h["_id"], h["_score"]) for h in bm25_resp["hits"]["hits"]]

        # ── kNN ───────────────────────────────────────────────────────────────
        knn_resp = client.search(
            index=INDEX_NAME,
            body={"query": _knn_query(vector, top_k), "size": top_k, "_source": False},
        )
        knn_hits = [(h["_id"], h["_score"]) for h in knn_resp["hits"]["hits"]]

        # ── RRF fusion (k=60) ─────────────────────────────────────────────────
        rrf_k = 60
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(knn_hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # keep top_k by fused score
        top_results = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        run[qid] = top_results
        
        # Debug output
        if verbose and query_count % 5 == 0:  # Print every 5th query to avoid too much output
            print(f"\n[QUERY {query_count}/{len(qid_to_question)}] {question}")
            
            # Ground truth
            if qrels and qid in qrels:
                relevant_docs = [doc_id for doc_id, rel in qrels[qid].items() if rel > 0]
                print(f"  Ground truth (relevant docs): {len(relevant_docs)} docs")
                if relevant_docs:
                    print(f"    Sample: {relevant_docs[:3]}")
            
            # Hybrid search results
            print(f"  Hybrid search results (top {len(top_results)}):")
            for i, (doc_id, score) in enumerate(list(top_results.items())[:5], 1):
                is_relevant = ""
                if qrels and qid in qrels and doc_id in qrels[qid]:
                    is_relevant = " ✓ RELEVANT" if qrels[qid][doc_id] > 0 else " ✗ not relevant"
                print(f"    {i}. {doc_id} (score={score:.4f}){is_relevant}")

    return run

#%%

# ── evaluation ────────────────────────────────────────────────────────────────

def _extract_doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Convert chunk doc_id to paragraph doc_id for matching qrels.
    
    Chunk IDs from indexing: "{title}-chunk-{chunk_idx}"
    Paragraph IDs in qrels:   "{title_slug}__p{para_idx}"
    
    This function extracts the title and maps to paragraph ID format.
    """
    # Chunk ID format: "ContractTitle-chunk-0"
    # Remove "-chunk-N" suffix to get the title
    parts = chunk_id.rsplit("-chunk-", 1)
    if len(parts) == 2:
        title = parts[0]
        # Convert title to slug format (same as _make_doc_id in cuad_dataset_utils)
        slug = title.replace(" ", "_").replace("/", "-")[:80]
        # All chunks from same title map to the same paragraph (p0)
        return f"{slug}__p0"
    return None


def _build_answer_spans(contracts: list[dict]) -> dict[str, list[dict]]:
    """
    Build a mapping from document title to list of answer spans.
    
    Returns
    -------
    dict: { title: [{ question, answer_text, answer_start }, ...] }
    """
    answer_spans: dict[str, list[dict]] = {}
    
    if not contracts:
        return answer_spans
    
    for contract in contracts:
        title = contract.get("title", "unknown")
        if title not in answer_spans:
            answer_spans[title] = []
        
        for para in contract.get("paragraphs", []):
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                
                for answer in answers:
                    answer_text = answer.get("text", "")
                    answer_start = answer.get("answer_start", -1)
                    
                    if answer_text and answer_start >= 0:
                        answer_spans[title].append({
                            "question": question,
                            "answer_text": answer_text,
                            "answer_start": answer_start,
                            "answer_end": answer_start + len(answer_text),
                        })
    
    return answer_spans


def _is_answer_in_chunk(answer_start: int, answer_end: int, chunk_start: int, chunk_end: int) -> bool:
    """
    Check if answer span overlaps with chunk boundaries.
    
    Returns True if the answer falls within [chunk_start, chunk_end).
    """
    return chunk_start <= answer_start < chunk_end and chunk_start < answer_end <= chunk_end


def _chunk_id_to_title(chunk_id: str) -> str:
    """
    Extract the title from a chunk ID.
    
    Chunk ID format: "{title}-chunk-{chunk_idx}"
    """
    parts = chunk_id.rsplit("-chunk-", 1)
    if len(parts) == 2:
        return parts[0]
    return chunk_id



def evaluate(qrels: dict, run: dict, contracts: list[dict] = None, qid_to_question: dict[str, str] = None, client = None) -> dict[str, float]:
    """
    Run pytrec_eval against qrels and run.
    Only queries present in both qrels and run are evaluated.
    Handles doc_id mismatches by mapping chunk IDs to paragraph IDs.
    
    Parameters
    ----------
    qrels : dict[str, dict[str, int]]
        Ground truth relevance judgments
    run : dict[str, dict[str, float]]
        Retrieved results from hybrid search
    contracts : list[dict], optional
        CUAD contracts data to check answer spans against chunks
    qid_to_question : dict[str, str], optional
        Mapping of query IDs to questions for debugging
    client : OpenSearch client, optional
        OpenSearch client for fetching chunk metadata
    
    Returns
    -------
    tuple[dict, dict]
        Aggregated metrics and per-query metrics
    """
    print(f"\n[DEBUG] qrels: {len(qrels)} queries, {sum(len(v) for v in qrels.values())} (qid, doc) pairs")
    # Build answer spans from contracts if provided
    answer_spans = _build_answer_spans(contracts) if contracts else {}
    
    if answer_spans:
        print(f"[INFO] Built answer spans for {len(answer_spans)} documents")
    
    print(f"[DEBUG] run: {len(run)} queries, {sum(len(v) for v in run.values())} (qid, doc) pairs")
    
    # Sample doc_id from qrels and run to detect mismatch
    sample_qrel_docid = None
    sample_run_docid = None
    for qid, docs in qrels.items():
        if docs:
            sample_qrel_docid = list(docs.keys())[0]
            break
    for qid, docs in run.items():
        if docs:
            sample_run_docid = list(docs.keys())[0]
            break
    
    print(f"[DEBUG] Sample qrel doc_id: {sample_qrel_docid}")
    print(f"[DEBUG] Sample run doc_id: {sample_run_docid}")
    
    # Map chunk IDs from run to paragraph IDs for qrels matching
    # run has doc_ids like "ContractTitle-chunk-0"
    # qrels has doc_ids like "contract_title__p0"
    # Also verify that answer spans fall within chunk boundaries if contract data is available
    mapped_run: dict[str, dict[str, float]] = {}
    mismatch_detected = False
    
    # We'll need to fetch chunk metadata from OpenSearch if we want to verify answers
    chunk_metadata_cache: dict[str, dict] = {}  # {chunk_id: {char_start, char_end, title}}
    
    for qid, docs in run.items():
        mapped_docs: dict[str, float] = {}
        
        # Get the question text for this query
        question = qid_to_question.get(qid, "") if qid_to_question else ""
        
        for doc_id, score in docs.items():
            mapped_doc_id = _extract_doc_id_from_chunk_id(doc_id)
            
            # If we have answer spans, verify the answer is in this chunk
            should_include = True
            if answer_spans and mapped_doc_id:
                title = _chunk_id_to_title(doc_id)
                
                # Fetch chunk metadata if not cached
                if doc_id not in chunk_metadata_cache:
                    try:
                        if client is None:
                            from cuad_opensearch.notebooks.open_search_connect import connect
                            temp_client = connect()
                        else:
                            temp_client = client
                        
                        chunk_doc = temp_client.get(index=INDEX_NAME, id=doc_id)
                        source = chunk_doc["_source"]
                        chunk_metadata_cache[doc_id] = {
                            "char_start": source.get("char_start", 0),
                            "char_end": source.get("char_end", 0),
                            "title": source.get("title", title)
                        }
                    except Exception as e:
                        print(f"[WARNING] Could not fetch metadata for chunk {doc_id}: {e}")
                        chunk_metadata_cache[doc_id] = {"char_start": -1, "char_end": -1, "title": title}
                
                metadata = chunk_metadata_cache[doc_id]
                chunk_start = metadata["char_start"]
                chunk_end = metadata["char_end"]
                title = metadata["title"]
                
                # Check if any answer for this question is within this chunk
                if title in answer_spans:
                    answer_found_in_chunk = False
                    for span in answer_spans[title]:
                        if span["question"] == question:
                            if _is_answer_in_chunk(span["answer_start"], span["answer_end"], chunk_start, chunk_end):
                                answer_found_in_chunk = True
                                break
                    should_include = answer_found_in_chunk
            
            if should_include and mapped_doc_id:
                # Aggregate scores if same paragraph appears multiple times (from different chunks)
                if mapped_doc_id in mapped_docs:
                    mapped_docs[mapped_doc_id] = max(mapped_docs[mapped_doc_id], score)
                else:
                    mapped_docs[mapped_doc_id] = score
                mismatch_detected = True
            elif should_include:
                # Fallback: use original doc_id
                mapped_docs[doc_id] = score
        
        # Keep top_k after mapping
        mapped_run[qid] = dict(
            sorted(mapped_docs.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        )
    
    if mismatch_detected:
        print("[INFO] Doc_id mismatch detected and mapped from chunk IDs to paragraph IDs")
        if answer_spans:
            print("[INFO] Answer spans verified against chunk boundaries")
        run = mapped_run
    
    # Filter run to queries that have at least one relevant doc in qrels
    qids_with_relevant = {
        qid for qid, docs in qrels.items() if any(r > 0 for r in docs.values())
    }
    print(f"[DEBUG] qids_with_relevant: {len(qids_with_relevant)}")
    
    filtered_run  = {qid: run[qid]  for qid in qids_with_relevant if qid in run}
    filtered_qrels = {qid: qrels[qid] for qid in filtered_run}
    
    print(f"[DEBUG] filtered_run: {len(filtered_run)} queries")
    print(f"[DEBUG] filtered_qrels: {len(filtered_qrels)} queries")
    
    if not filtered_run:
        print("[WARNING] No queries in filtered_run! Results will be 0.")
        return {metric: 0.0 for metric in METRICS}, {}

    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, METRICS)
    per_query  = evaluator.evaluate(filtered_run)

    # Macro-average across queries
    agg: dict[str, float] = {}
    for metric in METRICS:
        values = [per_query[qid][metric] for qid in per_query if metric in per_query[qid]]
        agg[metric] = sum(values) / len(values) if values else 0.0

    return agg, per_query


# ── main ──────────────────────────────────────────────────────────────────────
#%%
def main():
    client          = connect()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # 1. Load contracts from local CUAD JSON file
    json_path = _get_cuad_json_path()
    contracts = load_cuad_local_json(json_path)
    print(f"[DEBUG] Loaded {len(contracts)} contracts")
    
    # 2. Build ground truth (qrels and question mappings)
    qrels = build_qrels(contracts)
    qid_to_question = build_queries(contracts)
    print(f"[DEBUG] Built qrels with {len(qrels)} queries")
    print(f"[DEBUG] Built qid_to_question with {len(qid_to_question)} queries")
    
    # Sample qrels to see structure
    sample_qid = list(qrels.keys())[0] if qrels else None
    if sample_qid:
        print(f"[DEBUG] Sample qrel entry: qid='{sample_qid}', docs={list(qrels[sample_qid].items())[:3]}")

    # 3. Run hybrid search for every query
    run = hybrid_search_run(client, embedding_model, qid_to_question, qrels=qrels, top_k=TOP_K, verbose=True)
    print(f"[DEBUG] Run has {len(run)} queries")
    
    # Sample run to see structure
    sample_run_qid = list(run.keys())[0] if run else None
    if sample_run_qid:
        print(f"[DEBUG] Sample run entry: qid='{sample_run_qid}', docs={list(run[sample_run_qid].items())[:3]}")

    # 4. Evaluate
    agg_metrics, per_query = evaluate(qrels, run, contracts=contracts, qid_to_question=qid_to_question, client=client)


    # 5. Print results
    print("\n====== EVALUATION RESULTS (macro-averaged) ======")
    for metric, value in sorted(agg_metrics.items()):
        print(f"  {metric:<20} {value:.4f}")

    # Optional: save per-query breakdown
    out_path = Path("./eval_results_pytrec.json")
    with out_path.open("w") as f:
        json.dump({"aggregate": agg_metrics, "per_query": per_query}, f, indent=2)
    print(f"\nPer-query results saved to {out_path}")


if __name__ == "__main__":
    main()
# %%
