"""
evaluate_pdf_retrieval.py
─────────────────────────
Evaluate retrieval quality against CUAD Q&A ground truth.

Works with both index types:
    cuad_pdf_dataset  – chunks sourced from PDF extraction (default)
    cuad_dataset      – chunks sourced from CUAD_v1.json extraction

For every answered Q&A pair in CUAD_v1.json the script:
  1. Runs retrieval inside the contract (BM25 / KNN / Hybrid).
  2. Checks whether the ground-truth answer text appears in any returned chunk
     using whitespace-normalised substring matching.
  3. For the pdf index additionally verifies using mapped char offsets.

Metrics:  Hit@1, Hit@3, Hit@5, Hit@10, MRR@10, answered / total

Environment variables (all optional)
─────────────────────────────────────
  INDEX_NAME      target index            (default: cuad_pdf_dataset)
  EVAL_MODE       bm25 | knn | hybrid     (default: hybrid)
  TOP_K           candidates to retrieve  (default: 10)
  MAX_CONTRACTS   stop after N contracts  (0 = all)  (default: 0)
  MAX_QAS         Q&A pairs per contract  (0 = all)  (default: 0)
  RESULTS_FILE    write JSON report to    (default: eval_results.json)

Usage
─────
    python evaluate_pdf_retrieval.py                         # full hybrid run
    EVAL_MODE=bm25 python evaluate_pdf_retrieval.py          # BM25 only
    MAX_CONTRACTS=5 python evaluate_pdf_retrieval.py         # smoke test (5 docs)
    INDEX_NAME=cuad_dataset python evaluate_pdf_retrieval.py # JSON index
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import bisect
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_search_connect import connect  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
INDEX_NAME     = os.getenv("INDEX_NAME",  "cuad_pdf_dataset")
EVAL_MODE      = os.getenv("EVAL_MODE",   "hybrid").lower()   # bm25 | knn | hybrid
TOP_K          = int(os.getenv("TOP_K",   "10"))
MAX_CONTRACTS  = int(os.getenv("MAX_CONTRACTS", "0"))   # 0 = no limit
MAX_QAS        = int(os.getenv("MAX_QAS",  "0"))        # 0 = no limit per contract
RESULTS_FILE   = os.getenv("RESULTS_FILE", "eval_results.json")

CUAD_JSON = Path(__file__).resolve().parents[1] / "cuad_data" / "CUAD_v1" / "CUAD_v1.json"
PDF_ROOT  = Path(__file__).resolve().parents[1] / "cuad_data" / "CUAD_v1" / "full_contract_pdf"

IS_PDF_INDEX = "pdf" in INDEX_NAME.lower()

print(f"Index      : {INDEX_NAME}  ({'PDF' if IS_PDF_INDEX else 'JSON'} source)")
print(f"Mode       : {EVAL_MODE}   Top-K={TOP_K}")

# ── Connect & load model ──────────────────────────────────────────────────────
client = connect()
client.info()

embedding_model = None
if EVAL_MODE in ("knn", "hybrid"):
    print("Loading embedding model …")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("Model loaded.")

# ── Text utilities ────────────────────────────────────────────────────────────
def _ws_norm(text: str) -> str:
    """Collapse any whitespace run (incl. non-breaking) to a single space."""
    return re.sub(r"[\s\u00a0]+", " ", text).strip()


def _contains_answer(chunk_text: str, answer_text: str) -> bool:
    """True if answer_text (whitespace-normalised) appears inside chunk_text."""
    return _ws_norm(answer_text) in _ws_norm(chunk_text)


# ── Search functions (no OpenSearch neural plugin required) ───────────────────
def _bm25_search(query: str, title: str) -> list[dict]:
    body = {
        "query": {
            "bool": {
                "must":   [{"match": {"text": query}}],
                "filter": [{"term": {"title": title}}],
            }
        },
        "size": TOP_K,
        "_source": ["title", "text", "char_start", "char_end",
                    "page_start", "page_end", "page_char_start", "page_char_end"],
    }
    resp = client.search(index=INDEX_NAME, body=body, request_timeout=60)
    return [{"id": h["_id"], **h["_source"], "score": h["_score"]}
            for h in resp["hits"]["hits"]]


def _knn_search(query: str, title: str) -> list[dict]:
    vec = embedding_model.encode([query], normalize_embeddings=False)[0].tolist()
    body = {
        "query": {
            "bool": {
                "must": [{
                    "knn": {
                        "embedding": {"vector": vec, "k": TOP_K}
                    }
                }],
                "filter": [{"term": {"title": title}}],
            }
        },
        "size": TOP_K,
        "_source": ["title", "text", "char_start", "char_end",
                    "page_start", "page_end", "page_char_start", "page_char_end"],
    }
    resp = client.search(index=INDEX_NAME, body=body, request_timeout=60)
    return [{"id": h["_id"], **h["_source"], "score": h["_score"]}
            for h in resp["hits"]["hits"]]


def _hybrid_search(query: str, title: str) -> list[dict]:
    """Client-side RRF fusion of BM25 + KNN results."""
    bm25_results = _bm25_search(query, title)
    knn_results  = _knn_search(query, title)

    # Index by doc id for fast lookup
    all_docs: dict[str, dict] = {}
    for r in bm25_results + knn_results:
        all_docs[r["id"]] = r

    # RRF
    RRF_K = 60
    scores: dict[str, float] = {}
    for rank, r in enumerate(bm25_results, 1):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank)
    for rank, r in enumerate(knn_results, 1):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return [{**all_docs[doc_id], "score": rrf_score} for doc_id, rrf_score in ranked]


def search(query: str, title: str) -> list[dict]:
    if EVAL_MODE == "bm25":
        return _bm25_search(query, title)
    elif EVAL_MODE == "knn":
        return _knn_search(query, title)
    else:
        return _hybrid_search(query, title)


# ── JSON → PDF offset mapper (used for IS_PDF_INDEX validation) ───────────────
import difflib

def _tokenise(text: str) -> list[tuple[str, int]]:
    return [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]


def build_offset_mapper(json_text: str, pdf_text: str):
    """
    Return a closure map(json_off) -> pdf_off using word-level alignment.
    Result is cached per (json_text, pdf_text) pair via the closure.
    """
    j_toks = _tokenise(json_text)
    p_toks = _tokenise(pdf_text)
    sm = difflib.SequenceMatcher(None,
                                  [t[0] for t in j_toks],
                                  [t[0] for t in p_toks],
                                  autojunk=True)

    j_anchors = [0]
    p_anchors = [0]
    for blk in sm.get_matching_blocks():
        if blk.size == 0:
            continue
        for k in range(blk.size):
            jt = j_toks[blk.a + k]
            pt = p_toks[blk.b + k]
            j_anchors.append(jt[1])
            p_anchors.append(pt[1])
            j_anchors.append(jt[1] + len(jt[0]))
            p_anchors.append(pt[1] + len(pt[0]))
    j_anchors.append(len(json_text))
    p_anchors.append(len(pdf_text))

    SNAP = 200

    def _map(json_off: int, answer_text: str = "") -> tuple[int, bool]:
        idx = bisect.bisect_right(j_anchors, json_off) - 1
        idx = max(0, min(idx, len(j_anchors) - 2))
        j0, j1 = j_anchors[idx], j_anchors[idx + 1]
        p0, p1 = p_anchors[idx], p_anchors[idx + 1]
        candidate = p0 if j1 == j0 else int(p0 + (json_off - j0) / (j1 - j0) * (p1 - p0))

        if not answer_text:
            return candidate, False

        lo = max(0, candidate - SNAP)
        hi = min(len(pdf_text), candidate + SNAP + len(answer_text))
        window = pdf_text[lo:hi]
        # Exact
        idx2 = window.find(answer_text)
        if idx2 >= 0:
            return lo + idx2, True
        # Case-insensitive
        idx2 = window.lower().find(answer_text.lower())
        if idx2 >= 0:
            return lo + idx2, True
        # Whitespace-normalised
        norm_a = _ws_norm(answer_text)
        norm_w = _ws_norm(window)
        idx2 = norm_w.find(norm_a)
        if idx2 < 0:
            idx2 = norm_w.lower().find(norm_a.lower())
        if idx2 >= 0:
            toks = list(re.finditer(r"\S+", window))
            ntoks = list(re.finditer(r"\S+", norm_w))
            ti = bisect.bisect_right([t.start() for t in ntoks], idx2) - 1
            ti = max(0, min(ti, len(toks) - 1))
            return lo + toks[ti].start(), True
        return candidate, False

    return _map


# ── PDF text cache (avoid re-extracting the same PDF) ─────────────────────────
_pdf_text_cache: dict[str, str] = {}   # title -> full concatenated PDF text


def get_pdf_text(title: str) -> str | None:
    if title in _pdf_text_cache:
        return _pdf_text_cache[title]
    candidates = [
        p for p in PDF_ROOT.rglob("*")
        if p.suffix.upper() == ".PDF" and title.upper() in p.stem.upper()
    ]
    if not candidates:
        return None
    pdf_path = candidates[0]
    try:
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pg in pdf.pages:
                pages.append(pg.extract_text() or "")
        text = "\n\n".join(pages)
        _pdf_text_cache[title] = text
        return text
    except Exception:
        return None


# ── Offset mapper cache ───────────────────────────────────────────────────────
_mapper_cache: dict[str, object] = {}  # title -> map() closure


def get_mapper(title: str, json_text: str):
    if title in _mapper_cache:
        return _mapper_cache[title]
    pdf_text = get_pdf_text(title)
    if pdf_text is None:
        return None
    fn = build_offset_mapper(json_text, pdf_text)
    _mapper_cache[title] = fn
    return fn


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate():
    print(f"\nLoading {CUAD_JSON.name} …")
    data = json.loads(CUAD_JSON.read_text())
    contracts = data["data"]
    if MAX_CONTRACTS > 0:
        contracts = contracts[:MAX_CONTRACTS]

    hit_ranks: list[int | None] = []     # None = miss, int = 1-based rank of first hit
    skipped = 0
    per_contract: list[dict] = []

    for contract in tqdm(contracts, desc="Contracts", unit="contract"):
        title    = contract["title"]
        json_ctx = contract["paragraphs"][0]["context"]
        qas      = contract["paragraphs"][0]["qas"]
        answered = [q for q in qas if q["answers"]]
        if MAX_QAS > 0:
            answered = answered[:MAX_QAS]

        if not answered:
            continue

        contract_hits: list[dict] = []
        contract_miss = 0

        for qa in answered:
            question    = qa["question"]
            answer_obj  = qa["answers"][0]
            answer_text = answer_obj["text"]
            json_off    = answer_obj["answer_start"]

            # ── Run retrieval ────────────────────────────────────────────────
            try:
                results = search(question, title)
            except Exception as exc:
                skipped += 1
                continue

            if not results:
                hit_ranks.append(None)
                contract_miss += 1
                continue

            # ── Check each chunk for the answer ──────────────────────────────
            hit_rank = None
            for rank, chunk in enumerate(results, 1):
                if _contains_answer(chunk.get("text", ""), answer_text):
                    hit_rank = rank
                    break

            # ── Offset-based secondary check (PDF index only) ─────────────────
            offset_hit_rank = None
            if IS_PDF_INDEX and hit_rank is None:
                mapper = get_mapper(title, json_ctx)
                if mapper is not None:
                    pdf_off, _ = mapper(json_off, answer_text)
                    for rank, chunk in enumerate(results, 1):
                        if (chunk.get("char_start", 0) <= pdf_off
                                < chunk.get("char_end", 0)):
                            offset_hit_rank = rank
                            break
                    if offset_hit_rank is not None:
                        hit_rank = offset_hit_rank

            hit_ranks.append(hit_rank)
            if hit_rank is None:
                contract_miss += 1

            contract_hits.append({
                "question":    question,
                "answer":      answer_text,
                "hit_rank":    hit_rank,
                "top_result":  results[0].get("text", "")[:120] if results else "",
                "top_page":    results[0].get("page_start") if results else None,
            })

        per_contract.append({
            "title":     title,
            "total_qas": len(answered),
            "hits":      sum(1 for h in contract_hits if h["hit_rank"] is not None),
            "misses":    contract_miss,
            "detail":    contract_hits,
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total   = len(hit_ranks)
    answered_count = total - skipped

    def hit_at(k):
        return sum(1 for r in hit_ranks if r is not None and r <= k) / max(total, 1)

    def mrr(cutoff=10):
        return sum(1.0 / r for r in hit_ranks if r is not None and r <= cutoff) / max(total, 1)

    metrics = {
        "index":         INDEX_NAME,
        "mode":          EVAL_MODE,
        "top_k":         TOP_K,
        "timestamp":     datetime.now().isoformat(),
        "total_qas":     total,
        "skipped":       skipped,
        "hit_at_1":      round(hit_at(1),  4),
        "hit_at_3":      round(hit_at(3),  4),
        "hit_at_5":      round(hit_at(5),  4),
        "hit_at_10":     round(hit_at(10), 4),
        "mrr_at_10":     round(mrr(10),    4),
        "per_contract":  per_contract,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════╗")
    print(f"║  EVALUATION RESULTS — {EVAL_MODE.upper():<13}  ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  Index       : {INDEX_NAME:<22} ║")
    print(f"║  Total Q&A   : {total:<22} ║")
    print(f"║  Skipped     : {skipped:<22} ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  Hit@1       : {metrics['hit_at_1']:<22.4f} ║")
    print(f"║  Hit@3       : {metrics['hit_at_3']:<22.4f} ║")
    print(f"║  Hit@5       : {metrics['hit_at_5']:<22.4f} ║")
    print(f"║  Hit@10      : {metrics['hit_at_10']:<22.4f} ║")
    print(f"║  MRR@10      : {metrics['mrr_at_10']:<22.4f} ║")
    print("╚══════════════════════════════════════╝")

    # ── Write JSON report ─────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / RESULTS_FILE
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nDetailed report → {out_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
