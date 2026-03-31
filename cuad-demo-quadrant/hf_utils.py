"""
hf_utils.py
───────────
HuggingFace Hub utilities for document storage and URL generation.

Replaces s3_utils.py / MinIO with HuggingFace Hub dataset repository access.
PDFs are stored in a dataset repo under the prefix  raw/{title}.pdf
"""

import logging
import os
import time
from typing import Optional

from huggingface_hub import HfApi, hf_hub_url, login

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "ginntonicfun/cuad-pdf-contracts")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
# Branch / revision to resolve files from (default: main)
HF_REVISION = os.getenv("HF_REVISION", "main")

# Cached HfApi instance
_hf_api: Optional[HfApi] = None


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_hf_client() -> HfApi:
    """
    Initialise and cache the HfApi client.

    Authenticates with HF_TOKEN when present (required for private repos).
    Returns the cached HfApi instance.
    """
    global _hf_api

    if _hf_api is None:
        if HF_TOKEN:
            login(token=HF_TOKEN, add_to_git_credential=False)
            logger.info("HuggingFace Hub: authenticated as token owner")
        else:
            logger.warning("HF_TOKEN not set — only public repos will be accessible")

        _hf_api = HfApi(token=HF_TOKEN)
        logger.info("HuggingFace Hub client initialised (repo: %s)", HF_REPO_ID)

    return _hf_api


def get_hf_client() -> HfApi:
    """Return the cached HfApi client, initialising it if needed."""
    if _hf_api is None:
        init_hf_client()
    return _hf_api


# ─────────────────────────────────────────────────────────────────────────────
# URL generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_hf_url(hf_path: str, revision: Optional[str] = None) -> Optional[str]:
    """
    Build a direct-download URL for a file stored in the HuggingFace Hub repo.

    The URL resolves via:
      https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{hf_path}

    For private repos the HF_TOKEN is appended as a query parameter so the
    caller can open the URL without extra authentication headers.

    Args:
        hf_path:  Path of the file inside the repo  (e.g. ``raw/MyContract.pdf``)
        revision: Git revision / branch to resolve from (defaults to HF_REVISION)

    Returns:
        Resolved URL string, or None if URL construction fails.
    """
    try:
        url = hf_hub_url(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type=HF_REPO_TYPE,
            revision=revision or HF_REVISION,
        )

        # Append token for private repos so the URL is directly usable
        if HF_TOKEN:
            url = f"{url}?token={HF_TOKEN}"

        return url
    except Exception as e:
        logger.warning("Could not generate HuggingFace Hub URL for %s: %s", hf_path, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Document listing
# ─────────────────────────────────────────────────────────────────────────────

def list_hf_documents(prefix: str = "raw/") -> list[dict]:
    """
    List all PDF documents stored in the HuggingFace Hub dataset repo.

    Args:
        prefix: Path prefix to filter by (default: ``raw/``)

    Returns:
        List of dicts, each containing:
          - ``title``    – document name without the ``.pdf`` extension
          - ``hf_path``  – full path inside the repo  (e.g. ``raw/MyContract.pdf``)
          - ``pdf_url``  – direct-download URL (token-authenticated when HF_TOKEN is set)
    """
    try:
        api = get_hf_client()
        documents = []

        _t0 = time.perf_counter()
        repo_files = api.list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
        )
        logger.info("HuggingFace Hub list_repo_files completed in %.2fs", time.perf_counter() - _t0)

        for file_path in repo_files:
            if not file_path.startswith(prefix):
                continue
            if not file_path.lower().endswith(".pdf"):
                continue

            # Derive a human-readable title from the filename
            name = file_path[len(prefix):]       # strip leading prefix
            if name.lower().endswith(".pdf"):
                name = name[:-4]

            pdf_url = generate_hf_url(file_path)

            documents.append({
                "title": name,
                "hf_path": file_path,
                "pdf_url": pdf_url,
            })

        documents.sort(key=lambda d: d["title"].lower())
        logger.info("Listed %d documents from HuggingFace Hub repo '%s'", len(documents), HF_REPO_ID)
        return documents

    except Exception as e:
        logger.error("Failed to list HuggingFace Hub documents: %s", e)
        return []
