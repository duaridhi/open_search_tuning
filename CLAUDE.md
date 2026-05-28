# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cuad-ai-demo** is a FastAPI semantic-search service for the **CUAD** (Contract Understanding Atticus Dataset) legal-contract corpus. It exposes REST endpoints over a Qdrant vector index and a HuggingFace-Inference-API-backed RAG chat.

- **Search**: vector similarity over chunked contract text in Qdrant.
- **Highlighting**: per-result sentence reranking using a HuggingFace reranker model.
- **Chat (RAG)**: top-k retrieval → grounded answer via HuggingFace Inference chat completions.
- **Storage**: PDF source files live in a HuggingFace Hub *dataset* repo; download URLs are minted per-result.
- **Deployment**: Docker image targeting HuggingFace Spaces (port 7860).

## Architecture

### Entry points
- [app.py](app.py) — **active** FastAPI app. Uses HuggingFace Inference API for embeddings, reranking, and chat. Imports search from [cuad-demo-quadrant/qdrant_search_hf.py](cuad-demo-quadrant/qdrant_search_hf.py).
- [app_minio.py](app_minio.py) — **legacy** variant. Uses a self-hosted embedding service ([qdrant_search.py](cuad-demo-quadrant/qdrant_search.py)) and MinIO/S3 for PDFs. Kept for reference; not the deployed code path.

### `cuad-demo-quadrant/` — search & RAG package
- [qdrant_search_hf.py](cuad-demo-quadrant/qdrant_search_hf.py) — embeds the query and queries Qdrant; for **each** returned chunk, splits text into sentences and scores **each sentence** against the query with a reranker via HF Inference API to produce highlights.
- [qdrant_search.py](cuad-demo-quadrant/qdrant_search.py) — legacy variant that delegates embedding + highlight to a separate `EMBEDDING_SERVICE_URL` HTTP service.
- [chat_hf.py](cuad-demo-quadrant/chat_hf.py) — RAG generation; builds a context block from retrieved chunks and calls HF chat completions (default `Qwen/Qwen3-235B-A22B:novita`).
- [qdrant_cluster_connect.py](cuad-demo-quadrant/qdrant_cluster_connect.py) — singleton `QdrantClient`. Supports Qdrant Cloud via `CLUSTER_URL` or self-hosted via `QDRANT_URL`.
- [hf_utils.py](cuad-demo-quadrant/hf_utils.py) — HuggingFace Hub dataset client. Generates direct-download URLs (`hf_hub_url`) for `raw/{title}.pdf` and lists repo files.
- [document_utils.py](cuad-demo-quadrant/document_utils.py) — scans Qdrant via `scroll` to enumerate unique documents and per-document chunk stats.
- [s3_utils.py](cuad-demo-quadrant/s3_utils.py) — legacy MinIO/S3 helpers (used only by `app_minio.py`).
- [ingest_cuad_hf_bucket.py](cuad-demo-quadrant/ingest_cuad_hf_bucket.py) / [cuad_download_utils.py](cuad-demo-quadrant/cuad_download_utils.py) — one-off ingestion pipeline: download CUAD, push PDFs to the HF dataset repo.
- [upload_to_qdrant.py](cuad-demo-quadrant/upload_to_qdrant.py) — embeds chunks and upserts them into the `cuad_contracts` collection.

### Qdrant collection
- Name: `cuad_contracts` (override with `QDRANT_COLLECTION`).
- Vector size: **384** (matches `sentence-transformers/all-MiniLM-L6-v2`).
- Distance: cosine.
- Payload fields: `doc_id`, `title`, `text`, `page_start`, `page_end`, `pdf_path`, `char_start`, `char_end`, `page_offset_start`, `page_offset_end`.

## Models in use

| Role | Model | Where |
| --- | --- | --- |
| Query embedding | `sentence-transformers/all-MiniLM-L6-v2` | [qdrant_search_hf.py:27](cuad-demo-quadrant/qdrant_search_hf.py#L27) |
| Sentence reranker (highlight) | `BAAI/bge-reranker-v2-m3` | [qdrant_search_hf.py:28](cuad-demo-quadrant/qdrant_search_hf.py#L28) |
| RAG chat | `Qwen/Qwen3-235B-A22B:novita` (env: `CHAT_MODEL`) | [chat_hf.py:24](cuad-demo-quadrant/chat_hf.py#L24) |

All three are invoked through `huggingface_hub.InferenceClient` and require `HF_TOKEN`.

## API endpoints (from [app.py](app.py))

- `GET /health` — collection status + point count.
- `GET /search?q=...&top_k=10&document_name=...&strategy=semantic_search` — vector search with per-result sentence highlights.
- `GET /documents` — unique documents in the collection, enriched with HF Hub download URLs.
- `GET /documents/{document_name}` — per-document chunk inventory.
- `POST /chat` — RAG: retrieve top-k passages, generate grounded answer.

All endpoints wrap the underlying blocking calls in `asyncio.to_thread(...)` with per-endpoint `asyncio.wait_for(...)` timeouts (env-configurable: `SEARCH_TIMEOUT`, `CHAT_TIMEOUT`, etc.).

## Development

### Quick start
```bash
# install deps
pip install -r requirements.txt

# run dev server (HF-backed code path)
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Docker
```bash
docker-compose up --build           # default compose file
docker-compose -f docker-compose.dev.yml up --build
docker-compose -f docker-compose.prod.yml up --build
```

The HF Space deployment uses the root [Dockerfile](Dockerfile) on port 7860.

### Tests
- [tests/verify_collection.py](tests/verify_collection.py) — sanity check that the Qdrant collection is reachable and populated.
- Run with: `python tests/verify_collection.py`.

## Environment variables

Read from `.env` next to [qdrant_cluster_connect.py](cuad-demo-quadrant/qdrant_cluster_connect.py) and [chat_hf.py](cuad-demo-quadrant/chat_hf.py). **Never read `.env*` files directly** — refer to source for variable names.

Key variables (names only — see source for usage):
- `HF_TOKEN` — required; HuggingFace Inference + Hub access.
- `HF_REPO_ID`, `HF_REPO_TYPE`, `HF_REVISION` — HF dataset repo holding PDFs.
- `QDRANT_URL` or `CLUSTER_URL` + `QDRANT_API_KEY`, `QDRANT_PORT` — Qdrant target.
- `QDRANT_COLLECTION` — defaults to `cuad_contracts`.
- `CHAT_MODEL` — overrides the default chat model.
- `LOG_LEVEL`, `INIT_QDRANT_TIMEOUT`, `INIT_HF_TIMEOUT`, `COLLECTION_STATS_TIMEOUT`, `SEARCH_TIMEOUT`, `DOCS_LIST_TIMEOUT`, `DOCS_DETAIL_TIMEOUT`, `CHAT_TIMEOUT` — operational tuning.

## Known performance hotspot

The dominant latency cost in `/search` is per-sentence reranking in [qdrant_search_hf.highlight_text()](cuad-demo-quadrant/qdrant_search_hf.py#L74): for each of the `top_k` results, the function splits the chunk into sentences and calls `InferenceClient.text_classification` **once per sentence**, sequentially. With `top_k=10` and ~15–25 sentences per chunk, that is 150–250 sequential HTTPS round-trips per request. See `readme_docs/` (and the latency analysis in chat) for the recommended remediation path.

## Editing conventions
- Prefer targeted `Edit` over full-file rewrites.
- When asked to read or modify specific files, do **not** spawn broad parallel explore agents — read the named files directly.
- Do not read or display `.env*` files.

## Useful Claude Code agents & skills for this repo

Agents:
- **Explore** — locating cross-file references (search/embedding/chat integration points).
- **general-purpose** — multi-step refactors that touch ingestion + search together.
- **Plan** — designing the latency-reduction work below before editing.

Skills:
- **/code-review** — pre-PR review of search-path changes.
- **/simplify** — apply small cleanups from the review.
- **/verify** and **/run** — launch the API and hit `/search` to confirm a perf change works end-to-end.
- **/fewer-permission-prompts** — trim repeated allow-prompts during long iterations.

Skills that **do not apply** here:
- **/claude-api** — the project uses the HuggingFace Inference SDK, not the Anthropic SDK.
- **keybindings-help**, **statusline-setup**, **schedule**, **loop** — unrelated to the codebase.
