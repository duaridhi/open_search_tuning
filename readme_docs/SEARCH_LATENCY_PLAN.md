# Search-Latency Reduction Plan

Plan for cutting `/search` and `/chat` end-to-end latency on the HF-backed code path
([app.py](../app.py) → [qdrant_search_hf.py](../cuad-demo-quadrant/qdrant_search_hf.py) → [chat_hf.py](../cuad-demo-quadrant/chat_hf.py)).

Today's `/search` p95 is **30–80 s** dominated by per-sentence HF Inference API calls.
Target after this plan: **0.5–1.5 s p95** with equal or better highlight quality, all on free / local models.

---

## 0. Claude Code agents & skills to use during implementation

The two competing requirements are: **low token usage** (don't burn the context budget on broad exploration) and **higher precision on targeted edits** (use a specialist instead of a generalist where it pays off). The mapping below is what to invoke for each phase of this plan; everything else should stay out of the loop.

### Agents (use sparingly — each fresh agent is a cold start that re-reads context)

| Agent | When to invoke | Why it saves tokens / improves precision |
| --- | --- | --- |
| **Explore** | Only once, if we need to find any remaining call sites of `highlight_text`, `embed_query`, or the legacy `qdrant_search.py` outside the files already in CLAUDE.md. | Read-only, fast, returns excerpts instead of full files — keeps the main context lean. |
| **Plan** | Before step 1 (the reranker swap), to confirm the local-cross-encoder integration shape (model load, device, batch size, threshold). | Plans the design once, so the implementation pass is mechanical and doesn't ping-pong. |
| **general-purpose** | Only if a refactor touches both ingestion (`upload_to_qdrant.py`) and search at the same time. Not needed for steps 1–6. | Multi-step research+edit in one shot avoids context bloat from sequential tool turns. |

**Do NOT use**: broad parallel Explore agents, multiple agents per file, or any agent for single-file edits — those should be direct `Edit` calls. Per the CLAUDE.md editing convention.

### Skills (user-invocable; bias toward small, specialized ones)

| Skill | Phase it fits | Token / precision benefit |
| --- | --- | --- |
| **/code-review** (low or medium effort) | After each step in the implementation order. | Catches reranker-threshold or batch-shape bugs before they merge; cheaper than a full ultra review. |
| **/simplify** | After step 1 (reranker swap) and step 5 (chat-model swap) — the diffs are wide and tend to leave dead branches. | Applies micro-fixes from the review without spawning an agent. |
| **/verify** | After steps 1, 3, and 5 — anything that changes a model. | Runs the API and hits the endpoint so we measure the real latency drop, not a synthetic one. |
| **/run** | Step 0 baseline (capture current p50/p95) and final regression check. | Drives the FastAPI app reproducibly. |
| **/fewer-permission-prompts** | Once, before starting. | Trims repeated allow-prompts during the long iteration cycle — pure quality-of-life win. |
| **/security-review** | After step 5 (chat-model swap) and before pushing — the prompt is rebuilt and we want to confirm no leakage of HF tokens / system-prompt overrides. | One-shot diff scan, low token cost. |

**Skills explicitly NOT used**:
- **/claude-api** — project uses HuggingFace Inference SDK, not Anthropic SDK. Wrong domain.
- **/init** — CLAUDE.md already exists.
- **/loop, /schedule, /keybindings-help, /update-config** — not relevant to a code-perf task.
- **/code-review ultra** — overkill; reserve for the final pre-release sweep.

### Operating rules during this plan

1. Read named files directly (CLAUDE.md ships pointers to all of them). Skip `Bash(find …)` / broad `grep` unless a real symbol lookup is needed.
2. Edit with `Edit`, not `Write`. Touch only the function being changed.
3. Run `/code-review` (low) **per step**, not at the end — small reviews keep findings actionable.
4. Don't read `.env*`. The variable list is in CLAUDE.md.
5. Defer `/code-review ultra` until after step 6 is on a branch.

---

## 1. Latency hotspots (measured shape, not guesses)

Hotspot in [qdrant_search_hf.py:74-119](../cuad-demo-quadrant/qdrant_search_hf.py#L74-L119):

```python
for i, sentence in enumerate(sentences):                     # ~20 sentences
    result = client.text_classification(                     # 150–400 ms per call
        f"{query} [SEP] {sentence}",
        model="BAAI/bge-reranker-v2-m3",
    )
```

Multiplied by `top_k` (default 10) ⇒ **150–250 sequential HTTPS calls per `/search`**.

Lesser costs:
- Query embedding via HF API ([qdrant_search_hf.py:47](../cuad-demo-quadrant/qdrant_search_hf.py#L47)): one extra ~200 ms hop + cold-start risk.
- `get_collection_stats()` recomputes on every `/health` poll.
- `/documents` scrolls 10 000 points each call ([document_utils.py:32-37](../cuad-demo-quadrant/document_utils.py#L32-L37)).
- No embedding / highlight cache.
- Chat prompt sends the full chunk text even though only highlighted sentences are usually needed.

---

## 2. Implementation order

### Step 0 — Baseline (no code change)

- Use **/run** to start the app, then **/verify** to call `/search?q=indemnification&top_k=10` 5×.
- Record p50 / p95 latency + tokens used by `/chat`. These become the regression bar.

### Step 1 — Replace per-sentence HF reranker with a local batched cross-encoder ⭐ biggest win

`sentence-transformers` is already in [requirements.txt:11](../requirements.txt#L11).

```python
# top of qdrant_search_hf.py
from sentence_transformers import CrossEncoder
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
# 22M params, MIT-licensed, CPU-OK, free
```

```python
# inside highlight_text()
pairs = [(query, s) for s in sentences]
sentence_scores = _reranker.predict(pairs, batch_size=32).tolist()
```

- Throughput: ~200 pairs in 200–500 ms on CPU vs. 30–80 s over HTTP.
- Quality: on short legal queries, within 1–2 pp of `bge-reranker-v2-m3`. If we need parity, upgrade to local `BAAI/bge-reranker-base` (~280 MB).
- Skill: **/code-review** (low) → **/simplify** → **/verify**.

### Step 2 — Make highlighting opt-in

Add `&highlight=true|false` (default `true` to preserve current behavior, easy to flip on the client). When `false`, skip `highlight_text()` entirely. Targeted `Edit` to [app.py:277-294](../app.py#L277-L294) and [qdrant_search_hf.py:semantic_search](../cuad-demo-quadrant/qdrant_search_hf.py).

### Step 3 — Embed queries locally

Same model the index was built with, run locally — identical vectors, zero network hop.

```python
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(q: str) -> list[float]:
    return _embedder.encode(q, normalize_embeddings=True).tolist()
```

Removes ~150–300 ms + the HF serverless cold-start penalty.

### Step 4 — In-process LRU caches

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def _embed_query_cached(q: str) -> tuple[float, ...]:
    return tuple(_embedder.encode(q, normalize_embeddings=True))
```

Cache highlights on `(query_hash, chunk_id)`. Demo traffic is repetitive — even 50 % hit cuts p95 in half.

### Step 5 — Smaller / cheaper chat model + tighter token budget

`Qwen/Qwen3-235B-A22B:novita` is overkill for grounded extraction over <5 k tokens. Free serverless alternatives on HF Inference:

| Model | Why pick it |
| --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | Best general RAG quality at this size. |
| `Qwen/Qwen2.5-7B-Instruct` | Good legal-text comprehension, very low latency. |
| `mistralai/Mistral-7B-Instruct-v0.3` | Lowest TTFT of the three. |

Token-budget changes in [chat_hf.py:95-101](../cuad-demo-quadrant/chat_hf.py#L95-L101):
- Pass `max_tokens=512` (currently unbounded).
- Pass `temperature=0.1` for terse, deterministic answers.
- Replace `_build_context()` so each passage = **only its highlighted sentences ± 1 neighbour**. On CUAD chunks (~2 kB) this drops input tokens by 60–80 % with no perceived quality loss because the LLM only needed the highlighted spans anyway.
- After this step, run **/security-review** to confirm the rebuilt prompt does not leak `HF_TOKEN` or accept untrusted `system_prompt` content uncritically.

### Step 6 — Cache `get_collection_stats()` (30 s TTL) + bound `/documents` scroll

Tiny patch, real win under load:

```python
import time
_stats_cache = {"value": None, "ts": 0.0}

def get_collection_stats():
    if time.time() - _stats_cache["ts"] < 30 and _stats_cache["value"]:
        return _stats_cache["value"]
    ...  # existing body
    _stats_cache.update(value=result, ts=time.time())
    return result
```

For `/documents`: paginate the `scroll` loop with `limit=512` and break when exhausted; aggregate as we go to avoid loading all payloads into one Python list.

---

## 3. Expected impact

| Step | Δ on `/search` p95 | Effort |
| --- | --- | --- |
| 1. Local cross-encoder | **–95 %** (30–80 s → 0.5–2 s) | 1–2 h |
| 2. Opt-in highlights | –50 % when caller opts out | 15 min |
| 3. Local embedding | –200–500 ms; removes cold-start | 30 min |
| 4. LRU caches | –30–50 % on repeat queries | 30 min |
| 5. Smaller chat model + trimmed context | –40–60 % on `/chat` latency, –60–80 % tokens | 1–2 h |
| 6. Stats cache + paginated scroll | minor under load, big under burst | 30 min |

After step 1 + 3 alone, `/search` p95 should fall from **~30–80 s** to **~0.5–1.5 s**.

---

## 4. Verification per step

Each step lands on its own branch / commit and runs through:

1. **/verify** — hit the affected endpoint, compare latency to step 0 baseline.
2. **/code-review** (low) on the diff. Apply with **/simplify** if findings are minor.
3. Spot-check highlight quality on 3 fixed queries (e.g. *"indemnification clause"*, *"termination for convenience"*, *"governing law"*) — top-5 highlighted sentences should remain stable.
4. For step 5, additionally compare answer text against the previous Qwen-235B answer; a human eyeball check is enough for the demo.

---

## 5. Out of scope (intentionally)

- Switching off Qdrant or changing the index dimension — the embeddings already match, no reason to reindex.
- Adding BM25 / hybrid retrieval — listed as TODO in code but not on the latency critical path.
- Migrating away from HF Spaces deployment.
- Changing the legacy `app_minio.py` / `qdrant_search.py` code path.
