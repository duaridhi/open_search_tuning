# ---------------------------------------------------------------------------
# cuad_download_utils.py
# ---------------------------------------------------------------------------
# Shared utility for downloading the CUAD dataset from HuggingFace.
#
# Importable by any script in the project:
#
#   from cuad_download_utils import (
#       download_cuad_dataset,
#       find_pdfs,
#       find_txts,
#       load_contracts_from_download,
#       CUAD_LOCAL_DIR,
#   )
#
# Two ingestion workflows share this module:
#   - cuad_opensearch/notebooks/02_ingest_cuad_documents.py  → OpenSearch
#   - s3_utils/ingest_cuad_s3.py                             → MinIO
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "theatticusproject/cuad"

# Default download location: <project_root>/cuad_data/
CUAD_LOCAL_DIR = Path(__file__).resolve().parent / "cuad_data"

# Path to the SQuAD-format JSON inside the downloaded dataset
_JSON_RELATIVE_PATH = "CUAD_v1/CUAD_v1.json"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_cuad_dataset(
    local_dir: Path | str | None = None,
    max_workers: int = 8,
) -> Path:
    """
    Download CUAD contract PDFs and TXT files from HuggingFace.

    ``snapshot_download`` without explicit ``allow_patterns`` relies on HF's
    default filtering which skips non-Parquet files. Specifying patterns
    ensures both ``full_contract_pdf`` and ``full_contract_txt`` are fetched.
    Skips files that already exist locally.

    Parameters
    ----------
    local_dir:
        Root directory to download into. Defaults to ``CUAD_LOCAL_DIR``.
    max_workers:
        Number of parallel download workers.

    Returns
    -------
    Path
        The resolved local directory containing the downloaded dataset.
    """
    local_dir = Path(local_dir or CUAD_LOCAL_DIR).resolve()
    print(f"Downloading CUAD PDFs + TXTs + JSON to {local_dir} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["*.pdf", "*.PDF", "*.txt", "*.TXT", "*.json"],
        max_workers=max_workers,
    )
    print(f"✓ Download complete: {local_dir}")
    return local_dir


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------

def find_pdfs(directory: Path | str) -> list[Path]:
    """
    Recursively find all PDF files under *directory*.

    Returns a sorted, de-duplicated list handling both *.pdf and *.PDF.
    """
    directory = Path(directory)
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        for p in directory.rglob(pattern):
            if p not in seen:
                seen.add(p)
                pdfs.append(p)
    return sorted(pdfs)


def find_txts(directory: Path | str) -> list[Path]:
    """
    Recursively find all TXT contract files under *directory*.

    Returns a sorted, de-duplicated list handling both *.txt and *.TXT.
    """
    directory = Path(directory)
    seen: set[Path] = set()
    txts: list[Path] = []
    for pattern in ("*.txt", "*.TXT"):
        for p in directory.rglob(pattern):
            if p not in seen:
                seen.add(p)
                txts.append(p)
    return sorted(txts)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_contracts_from_download(
    local_dir: Path | str | None = None,
) -> list[dict]:
    """
    Load CUAD contracts from the SQuAD-format JSON inside the downloaded
    dataset directory.

    Returns a list of contract dicts, each with:
        - ``title``      (str)  contract filename / ID
        - ``paragraphs`` (list) with a single ``{context: str, qas: [...]}``

    This gives one unique entry per contract (510 total), avoiding the
    ~41 duplicate-context rows that appear in the HuggingFace streaming API.

    Parameters
    ----------
    local_dir:
        Root directory of the downloaded dataset. Defaults to
        ``CUAD_LOCAL_DIR``.

    Returns
    -------
    list[dict]
        Contracts loaded from ``CUAD_v1.json``.

    Raises
    ------
    FileNotFoundError
        If the JSON file is not found — run ``download_cuad_dataset()`` first.
    """
    local_dir = Path(local_dir or CUAD_LOCAL_DIR).resolve()
    json_path = local_dir / _JSON_RELATIVE_PATH
    if not json_path.exists():
        raise FileNotFoundError(
            f"CUAD JSON not found at {json_path}.\n"
            "Run download_cuad_dataset() first."
        )
    with json_path.open() as f:
        data = json.load(f)
    contracts = data["data"]
    print(f"✓ Loaded {len(contracts)} contracts from {json_path}")
    return contracts
