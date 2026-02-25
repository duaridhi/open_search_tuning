"""
cuad_dataset_utils.py
---------------------
Generic utilities for loading and transforming the CUAD dataset into data
structures needed for indexing, search, and evaluation.

All functions are stateless and framework-agnostic — they only depend on
the HuggingFace `datasets` library and the Python standard library.

Public API — JSON / SQuAD-style source
---------------------------------------
load_cuad_hf(split, dataset_name)          → HuggingFace Dataset
load_cuad_local_json(json_path)            → list[dict]  (contracts)
iter_cuad_paragraphs(contracts)            → Iterator[dict]
build_corpus(contracts)                    → dict[doc_id, {title, text}]
build_queries(contracts)                   → dict[qid, question_text]
build_qrels(contracts)                     → dict[qid, {doc_id: relevance}]
build_qrels_hf(hf_dataset)                → (qrels, qid_to_question)
build_beir_triplet(json_path|contracts)    → (corpus, queries, qrels)
iter_index_actions(contracts, model, ...)  → Iterator[OpenSearch bulk action]

Public API — CSV + file-system source
---------------------------------------
load_cuad_csv(csv_path)                    → (categories, rows)
iter_contracts_from_csv(csv, txt, pdf)     → Iterator[dict]
build_contracts_dict(csv, txt, pdf)        → dict[doc_name, record]
iter_index_actions_from_csv(dict, model)   → Iterator[OpenSearch bulk action]

ID helpers
----------
question_to_qid(question)                  → str  (stable 8-char MD5 slug)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HF_DATASET = "theatticusproject/cuad-qa"
DEFAULT_JSON_FIELD = "data"          # top-level key inside CUAD_v1.json


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def question_to_qid(question: str) -> str:
    """
    Return a stable 8-char hex ID for a question string.

    Deterministic across runs so qrel/run dicts always align.
    """
    return hashlib.md5(question.strip().encode()).hexdigest()[:8]


def _make_doc_id(title: str, paragraph_index: int) -> str:
    """
    Construct a document ID from a contract title and paragraph index.
    Format: '<title_slug>__p<index>'
    """
    slug = title.replace(" ", "_").replace("/", "-")[:80]
    return f"{slug}__p{paragraph_index}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cuad_hf(
    split: str = "test",
    dataset_name: str = DEFAULT_HF_DATASET,
    *,
    trust_remote_code: bool = True,
):
    """
    Load a CUAD split directly from HuggingFace Hub.

    Parameters
    ----------
    split : str
        HuggingFace split name, e.g. ``"train"``, ``"test"``.
    dataset_name : str
        HuggingFace dataset repo, default ``"theatticusproject/cuad-qa"``.
    trust_remote_code : bool
        Passed through to ``load_dataset``.

    Returns
    -------
    datasets.Dataset
    """
    from datasets import load_dataset  # lazy import — keeps utils lightweight

    print(f"[cuad_utils] Loading '{dataset_name}' ({split}) from HuggingFace …")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)
    print(f"[cuad_utils]   {len(ds):,} samples loaded.")
    return ds


def load_cuad_local_json(
    json_path: str | Path,
    field: str = DEFAULT_JSON_FIELD,
):
    """
    Load CUAD from the local ``CUAD_v1.json`` file (SQuAD-style format).

    The JSON has the structure::

        { "data": [ { "title": "...", "paragraphs": [...] }, ... ] }

    Parameters
    ----------
    json_path : str or Path
        Absolute path to ``CUAD_v1.json``.
    field : str
        Top-level key that contains the list of contracts (default ``"data"``).

    Returns
    -------
    list[dict]
        Raw list of contract dicts, each with ``"title"`` and ``"paragraphs"``.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"CUAD JSON not found at: {json_path}")

    print(f"[cuad_utils] Loading local JSON from {json_path} …")
    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    contracts = raw[field]
    print(f"[cuad_utils]   {len(contracts):,} contracts loaded.")
    return contracts


# ---------------------------------------------------------------------------
# Iteration helpers (SQuAD-style local JSON)
# ---------------------------------------------------------------------------

def iter_cuad_paragraphs(contracts: list[dict]) -> Iterator[dict]:
    """
    Iterate over every paragraph in the SQuAD-style CUAD JSON.

    Yields
    ------
    dict with keys:
        doc_id   : str  — unique paragraph identifier
        title    : str  — contract title
        text     : str  — paragraph context text
        qas      : list — list of QA dicts for that paragraph
    """
    for contract in contracts:
        title = contract.get("title", "unknown")
        for p_idx, para in enumerate(contract.get("paragraphs", [])):
            yield {
                "doc_id": _make_doc_id(title, p_idx),
                "title":  title,
                "text":   para["context"],
                "qas":    para.get("qas", []),
            }


# ---------------------------------------------------------------------------
# BEIR-style data structures
# ---------------------------------------------------------------------------

def build_corpus(contracts: list[dict]) -> dict[str, dict]:
    """
    Build a BEIR-compatible corpus dict from SQuAD-style CUAD contracts.

    Returns
    -------
    corpus : { doc_id: {"title": str, "text": str} }
    """
    corpus: dict[str, dict] = {}
    for para in iter_cuad_paragraphs(contracts):
        corpus[para["doc_id"]] = {
            "title": para["title"],
            "text":  para["text"],
        }
    print(f"[cuad_utils] Corpus built: {len(corpus):,} passages.")
    return corpus


def build_queries(contracts: list[dict]) -> dict[str, str]:
    """
    Collect all unique questions from the CUAD dataset and assign stable IDs.

    Returns
    -------
    queries : { qid: question_text }
    """
    queries: dict[str, str] = {}
    for para in iter_cuad_paragraphs(contracts):
        for qa in para["qas"]:
            q = qa.get("question", "").strip()
            if q:
                queries[question_to_qid(q)] = q
    print(f"[cuad_utils] Queries built: {len(queries):,} unique questions.")
    return queries


def build_qrels(contracts: list[dict]) -> dict[str, dict[str, int]]:
    """
    Build pytrec_eval / BEIR-compatible qrels from SQuAD-style CUAD contracts.

    Relevance grades:
        1  — the paragraph contains at least one non-empty answer span
        0  — the paragraph has the question but no answer (unanswerable)

    Returns
    -------
    qrels : { qid: { doc_id: relevance } }
    """
    qrels: dict[str, dict[str, int]] = {}
    for para in iter_cuad_paragraphs(contracts):
        doc_id = para["doc_id"]
        for qa in para["qas"]:
            q = qa.get("question", "").strip()
            if not q:
                continue
            qid = question_to_qid(q)
            relevant = any(
                ans.get("text", "").strip()
                for ans in qa.get("answers", [])
            )
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = 1 if relevant else 0

    total_pairs = sum(len(v) for v in qrels.values())
    print(
        f"[cuad_utils] Qrels built: {len(qrels):,} queries, "
        f"{total_pairs:,} (query, doc) pairs."
    )
    return qrels


def build_qrels_hf(hf_dataset) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """
    Build qrels and qid→question mapping from a HuggingFace CUAD-QA dataset.

    Suitable for datasets loaded via ``load_cuad_hf()``.

    Returns
    -------
    qrels          : { qid: { doc_id: relevance } }
    qid_to_question: { qid: question_text }
    """
    qrels: dict[str, dict[str, int]] = {}
    qid_to_question: dict[str, str] = {}

    for sample in hf_dataset:
        qid    = question_to_qid(sample["question"])
        doc_id = sample["id"]
        relevant = len(sample["answers"]["text"]) > 0

        qid_to_question[qid] = sample["question"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = 1 if relevant else 0

    total_pairs = sum(len(v) for v in qrels.values())
    print(
        f"[cuad_utils] Qrels (HF) built: {len(qrels):,} queries, "
        f"{total_pairs:,} (query, doc) pairs."
    )
    return qrels, qid_to_question


# ---------------------------------------------------------------------------
# Combined convenience loader (BEIR triplet)
# ---------------------------------------------------------------------------

def build_beir_triplet(
    json_path: str | Path | None = None,
    contracts: list[dict] | None = None,
) -> tuple[dict, dict, dict]:
    """
    Return the standard BEIR triplet ``(corpus, queries, qrels)`` from either
    a local JSON path or a pre-loaded contracts list.

    Parameters
    ----------
    json_path : str or Path, optional
        Path to ``CUAD_v1.json``.  Used when ``contracts`` is ``None``.
    contracts : list[dict], optional
        Pre-loaded list returned by ``load_cuad_local_json()``.

    Returns
    -------
    corpus  : { doc_id: {"title": str, "text": str} }
    queries : { qid: question_text }
    qrels   : { qid: { doc_id: relevance } }
    """
    if contracts is None:
        if json_path is None:
            raise ValueError("Provide either 'json_path' or 'contracts'.")
        contracts = load_cuad_local_json(json_path)

    corpus  = build_corpus(contracts)
    queries = build_queries(contracts)
    qrels   = build_qrels(contracts)
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# OpenSearch bulk-indexing helper  (JSON / SQuAD-style source)
# ---------------------------------------------------------------------------

def iter_index_actions(
    contracts: list[dict],
    embedding_model,
    index_name: str,
    encode_batch_size: int = 32,
    max_docs: int | None = None,
    last_doc_id: str | None = None,
) -> Iterator[dict]:
    """
    Yield OpenSearch bulk-action dicts for every paragraph in ``contracts``.

    Performs batched embedding encoding and supports checkpoint-style resumption
    via ``last_doc_id``.

    Parameters
    ----------
    contracts        : list of contract dicts from ``load_cuad_local_json()``.
    embedding_model  : SentenceTransformer (or any model with ``.encode()``).
    index_name       : target OpenSearch index.
    encode_batch_size: number of texts encoded in one ``model.encode()`` call.
    max_docs         : stop after this many docs (``None`` = no limit).
    last_doc_id      : resume after this doc_id (skip everything before it).
    """
    batch_docs:  list[dict] = []
    batch_texts: list[str]  = []
    skip_mode = last_doc_id is not None
    total = 0

    def _flush():
        embeddings = embedding_model.encode(
            batch_texts,
            show_progress_bar=False,
            batch_size=encode_batch_size,
            normalize_embeddings=False,
        )
        for idx, doc in enumerate(batch_docs):
            yield {
                "_index": index_name,
                "_id":    doc["doc_id"],
                "_source": {
                    "doc_id":      doc["doc_id"],
                    "title":       doc["title"],
                    "text":        doc["text"],
                    "text_vector": embeddings[idx].tolist(),
                },
            }

    for para in iter_cuad_paragraphs(contracts):
        if max_docs is not None and total >= max_docs:
            break

        if skip_mode:
            if para["doc_id"] == last_doc_id:
                skip_mode = False
            continue

        batch_docs.append(para)
        batch_texts.append(para["text"])
        total += 1

        if len(batch_docs) >= encode_batch_size:
            yield from _flush()
            batch_docs.clear()
            batch_texts.clear()

    if batch_docs:
        yield from _flush()


# ---------------------------------------------------------------------------
# CSV-based utilities  (master clauses CSV + full_contracts_txt/pdf folders)
# ---------------------------------------------------------------------------
#
# CSV layout (83 columns, 511 rows):
#   col 0          : contract filename  (matches PDF/TXT basenames)
#   cols 1 … 82    : 41 category pairs — each pair = (clause/context text, answer)
#
# Each record yielded by the iterators has the shape:
#   {
#       doc_name : str,            # filename stem (no extension)
#       filename : str,            # original value from first CSV column
#       txt_path : Path | None,    # path to .txt file (if txt_dir supplied)
#       pdf_path : Path | None,    # path to .pdf file (if pdf_dir supplied)
#       text     : str | None,     # raw text read from .txt file
#       clauses  : {               # one entry per CUAD category
#           category_name : {
#               "context" : str,   # clause/context text from CSV
#               "answer"  : str,   # human-input answer from CSV
#           }
#       }
#   }
# ---------------------------------------------------------------------------

def load_cuad_csv(csv_path: str | Path) -> tuple[list[str], list[dict]]:
    """
    Load the CUAD master clauses CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the 83-column master clauses CSV (e.g. ``master_clauses.csv``).

    Returns
    -------
    categories : list[str]
        The 41 category names derived from the column headers.
    rows : list[dict]
        One dict per contract row, keyed by column header.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CUAD CSV not found at: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    headers = list(rows[0].keys()) if rows else []

    # Category columns are those that do NOT end with an answer-suffix.
    categories: list[str] = []
    for h in headers[1:]:
        h_lower = h.lower()
        if not (h_lower.endswith("-answer") or h_lower.endswith(" (answer)")):
            categories.append(h.strip())

    print(f"[cuad_utils] CSV loaded: {len(rows):,} contracts, {len(categories)} categories.")
    return categories, rows


def _resolve_paths(
    filename: str,
    txt_dir: Path | None,
    pdf_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    """Resolve TXT and PDF paths for a given contract filename stem."""
    stem = Path(filename).stem
    txt_path = (txt_dir / f"{stem}.txt") if txt_dir else None
    pdf_path = (pdf_dir / f"{stem}.pdf") if pdf_dir else None
    # Fallback: try the raw filename string as given in the CSV
    if txt_path and not txt_path.exists() and txt_dir:
        alt = txt_dir / filename
        if alt.exists():
            txt_path = alt
    if pdf_path and not pdf_path.exists() and pdf_dir:
        alt = pdf_dir / filename
        if alt.exists():
            pdf_path = alt
    return txt_path, pdf_path


def _parse_clauses(row: dict, all_headers: list[str]) -> dict[str, dict[str, str]]:
    """
    Extract {category → {context, answer}} from a single CSV row.

    Handles two common header patterns:
      Pattern A — alternating pairs: ``"Category"`` then ``"Category-Answer"``
      Pattern B — labelled pairs:    ``"Category (Context)"`` then ``"Category (Answer)"``
    """
    clauses: dict[str, dict[str, str]] = {}
    col_names = all_headers[1:]   # skip filename column

    i = 0
    while i < len(col_names):
        h       = col_names[i]
        h_lower = h.lower()

        if h_lower.endswith("-answer") or h_lower.endswith(" (answer)"):
            i += 1
            continue  # belongs to a preceding context column

        # Determine if the immediately next column is the answer pair
        answer_col: str | None = None
        if i + 1 < len(col_names):
            next_h = col_names[i + 1]
            if next_h.lower().endswith("-answer") or next_h.lower().endswith(" (answer)"):
                answer_col = next_h

        category = h.strip()
        clauses[category] = {
            "context": row.get(h, "").strip(),
            "answer":  row.get(answer_col, "").strip() if answer_col else "",
        }
        i += 2 if answer_col else 1

    return clauses


def iter_contracts_from_csv(
    csv_path: str | Path,
    txt_dir: str | Path | None = None,
    pdf_dir: str | Path | None = None,
    *,
    read_text: bool = True,
) -> Iterator[dict]:
    """
    Iterate over CUAD contracts using the master clauses CSV as the primary source.

    Parameters
    ----------
    csv_path  : path to the 83-column master clauses CSV.
    txt_dir   : directory containing ``.txt`` full-contract files (optional).
    pdf_dir   : directory containing ``.pdf`` full-contract files (optional).
    read_text : if ``True`` (default) and ``txt_dir`` is given, read raw text
                from the matching ``.txt`` file.

    Yields
    ------
    dict  — one per contract row (shape described in the section header).
    """
    txt_dir = Path(txt_dir) if txt_dir else None
    pdf_dir = Path(pdf_dir) if pdf_dir else None

    _categories, rows = load_cuad_csv(csv_path)

    if not rows:
        return

    all_headers = list(rows[0].keys())

    for row in rows:
        first_key = all_headers[0]
        filename  = row[first_key].strip()
        if not filename:
            continue

        doc_name           = Path(filename).stem
        txt_path, pdf_path = _resolve_paths(filename, txt_dir, pdf_dir)

        raw_text: str | None = None
        if read_text and txt_path and txt_path.exists():
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace")

        clauses = _parse_clauses(row, all_headers)

        yield {
            "doc_name": doc_name,
            "filename": filename,
            "txt_path": txt_path,
            "pdf_path": pdf_path,
            "text":     raw_text,
            "clauses":  clauses,
        }


def build_contracts_dict(
    csv_path: str | Path,
    txt_dir: str | Path | None = None,
    pdf_dir: str | Path | None = None,
    *,
    read_text: bool = True,
) -> dict[str, dict]:
    """
    Build a ``{doc_name: record}`` dict from the CUAD master clauses CSV.

    This is the primary entry-point for downstream indexing and evaluation
    when working with the CSV + file-system layout.

    Parameters
    ----------
    csv_path, txt_dir, pdf_dir, read_text
        Forwarded to ``iter_contracts_from_csv()``.

    Returns
    -------
    dict[str, dict]
        Keys are contract name stems; values are the per-contract dicts
        (doc_name, filename, txt_path, pdf_path, text, clauses).

    Example
    -------
    >>> contracts = build_contracts_dict(
    ...     "data/master_clauses.csv",
    ...     txt_dir="data/full_contracts_txt",
    ...     pdf_dir="data/full_contracts_pdf",
    ... )
    >>> rec = contracts["LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"]
    >>> print(rec["text"][:200])          # raw contract text
    >>> print(rec["pdf_path"])            # Path to PDF
    >>> print(rec["clauses"]["Document Name"]["answer"])
    """
    contracts = {
        rec["doc_name"]: rec
        for rec in iter_contracts_from_csv(
            csv_path, txt_dir, pdf_dir, read_text=read_text
        )
    }
    txt_found = sum(1 for v in contracts.values() if v["text"] is not None)
    pdf_found = sum(
        1 for v in contracts.values() if v["pdf_path"] and v["pdf_path"].exists()
    )
    print(
        f"[cuad_utils] contracts_dict: {len(contracts):,} contracts "
        f"| {txt_found} txt files read | {pdf_found} pdf paths resolved."
    )
    return contracts


def iter_index_actions_from_csv(
    contracts_dict: dict[str, dict],
    embedding_model,
    index_name: str,
    encode_batch_size: int = 32,
    max_docs: int | None = None,
    last_doc_id: str | None = None,
) -> Iterator[dict]:
    """
    Yield OpenSearch bulk-action dicts from a ``build_contracts_dict()`` result.

    The ``_source`` of each indexed document includes:
      - ``doc_id``, ``title`` (doc_name), ``text`` (full contract text)
      - ``pdf_path`` (string), ``clauses`` (category → context/answer dict)
      - ``text_vector`` (dense embedding of the full text)

    When the TXT file is absent, the text is assembled by concatenating all
    clause context strings from the CSV.

    Parameters
    ----------
    contracts_dict   : output of ``build_contracts_dict()``.
    embedding_model  : SentenceTransformer (or any model with ``.encode()``).
    index_name       : target OpenSearch index name.
    encode_batch_size: batch size for ``model.encode()``.
    max_docs         : cap on total docs yielded (``None`` = no limit).
    last_doc_id      : resume — skip all docs up to and including this id.
    """
    batch_ids:   list[str]  = []
    batch_texts: list[str]  = []
    batch_recs:  list[dict] = []
    skip_mode = last_doc_id is not None
    total = 0

    def _flush():
        embeddings = embedding_model.encode(
            batch_texts,
            show_progress_bar=False,
            batch_size=encode_batch_size,
            normalize_embeddings=False,
        )
        for idx, rec in enumerate(batch_recs):
            yield {
                "_index": index_name,
                "_id":    batch_ids[idx],
                "_source": {
                    "doc_id":      batch_ids[idx],
                    "title":       rec["doc_name"],
                    "text":        batch_texts[idx],
                    "pdf_path":    str(rec["pdf_path"]) if rec["pdf_path"] else None,
                    "clauses":     rec["clauses"],
                    "text_vector": embeddings[idx].tolist(),
                },
            }

    for doc_id, rec in contracts_dict.items():
        if max_docs is not None and total >= max_docs:
            break

        if skip_mode:
            if doc_id == last_doc_id:
                skip_mode = False
            continue

        # Prefer full TXT; fall back to concatenated clause contexts from CSV
        text = rec["text"] or " ".join(
            v["context"] for v in rec["clauses"].values() if v["context"]
        )
        if not text:
            continue

        batch_ids.append(doc_id)
        batch_texts.append(text)
        batch_recs.append(rec)
        total += 1

        if len(batch_ids) >= encode_batch_size:
            yield from _flush()
            batch_ids.clear()
            batch_texts.clear()
            batch_recs.clear()

    if batch_ids:
        yield from _flush()
