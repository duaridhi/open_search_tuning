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

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import pytrec_eval
from sentence_transformers import SentenceTransformer

# ── path setup so open_search_connect is importable ──────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cuad_opensearch.notebooks.open_search_connect import connect
from cuad_dataset_utils import (
    load_cuad_hf,
    build_qrels_hf,
)

load_dotenv(HERE.parent / ".env")

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
            "context": {
                "query": text,
                "analyzer": "english",
            }
        }
    }


def hybrid_search_run(
    client,
    embedding_model: SentenceTransformer,
    qid_to_question: dict[str, str],
    top_k: int = TOP_K,
) -> dict[str, dict[str, float]]:
    """
    Run hybrid search for every query and return a pytrec_eval run dict.

    Fusion strategy: Reciprocal Rank Fusion (RRF, k=60) — parameter-free,
    robust default for combining BM25 + dense rankings.

    Returns
    -------
    run : {qid: {doc_id: rrf_score}}
    """
    run: dict[str, dict[str, float]] = {}

    for qid, question in tqdm(qid_to_question.items(), desc="Searching"):
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
        run[qid] = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

    return run


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(qrels: dict, run: dict) -> dict[str, float]:
    """
    Run pytrec_eval against qrels and run.
    Only queries present in both qrels and run are evaluated.
    """
    # Filter run to queries that have at least one relevant doc in qrels
    qids_with_relevant = {
        qid for qid, docs in qrels.items() if any(r > 0 for r in docs.values())
    }
    filtered_run  = {qid: run[qid]  for qid in qids_with_relevant if qid in run}
    filtered_qrels = {qid: qrels[qid] for qid in filtered_run}

    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, METRICS)
    per_query  = evaluator.evaluate(filtered_run)

    # Macro-average across queries
    agg: dict[str, float] = {}
    for metric in METRICS:
        values = [per_query[qid][metric] for qid in per_query if metric in per_query[qid]]
        agg[metric] = sum(values) / len(values) if values else 0.0

    return agg, per_query


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    client          = connect()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # 1. Build ground truth (via cuad_dataset_utils)
    hf_dataset = load_cuad_hf(split="test")
    qrels, qid_to_question = build_qrels_hf(hf_dataset)

    # 2. Run hybrid search for every query
    run = hybrid_search_run(client, embedding_model, qid_to_question, top_k=TOP_K)

    # 3. Evaluate
    agg_metrics, per_query = evaluate(qrels, run)

    # 4. Print results
    print("\n====== EVALUATION RESULTS (macro-averaged) ======")
    for metric, value in sorted(agg_metrics.items()):
        print(f"  {metric:<20} {value:.4f}")

    # Optional: save per-query breakdown
    out_path = HERE.parent / "eval_results_pytrec.json"
    with out_path.open("w") as f:
        json.dump({"aggregate": agg_metrics, "per_query": per_query}, f, indent=2)
    print(f"\nPer-query results saved to {out_path}")


if __name__ == "__main__":
    main()