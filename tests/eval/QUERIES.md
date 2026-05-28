# Query Set Schema

`queries.json` is the canonical, append-only CUAD-derived query set used for
retrieval evaluation. It contains **82 entries**: the 41 CUAD clause
categories times 2 query forms (`short`, `question`). Schema per entry:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | `<category-slug>__<form>`. Stable. Used as the key in `gold.json` and in per-query run output. |
| `q` | string | The actual query text sent to `/search?q=...`. For `form="short"` this is the canonical CUAD category name (e.g. `"Governing Law"`); for `form="question"` this is the original CUAD question prompt verbatim. |
| `category` | string | Canonical CUAD category name. One of 41. Used by `build_gold.py` to look up annotated spans. |
| `form` | string | `"short"` or `"question"`. The `short` form is the headline metric (it is what a UI user types); `question` is a paraphrase-generalization signal. |
| `doc` | string \| null | Document-name filter. `null` for the corpus-wide partition. Doc-scoped runs are constructed at eval time by joining these query texts with the 5 titles in `doc_scoped_titles.json`; doc-scoped variants are not enumerated in this file. |

`queries_smoke.json` is a separate file with the 8 legacy short-form queries
from the agent's original system prompt. Same shape minus `category`/`form`.
Run with `run_eval.py --queries tests/eval/queries_smoke.json`.

**Append-only.** Existing entries must never be removed or reworded — that
would invalidate the baseline. New queries are appended at the end.
