# Evaluation Strategy for `/search` and `/chat` on the CUAD Corpus

Status: **proposal**. Nothing has been built yet. The user must approve the
"Build plan" and answer the "Open questions for the user" section before
`queries.json`, `gold.json`, or `run_eval.py` are created.

---

## 1. Background

The service under test indexes the **Contract Understanding Atticus Dataset**
(CUAD v1, Hendrycks et al. 2021, [arXiv:2103.06268](https://arxiv.org/abs/2103.06268)):
510 commercial contracts sourced from SEC EDGAR, with **~13,000 expert
annotations across 41 clause categories** (e.g. *Governing Law*,
*Change of Control*, *Limitation of Liability*).
Source-of-truth dataset card:
[`theatticusproject/cuad-qa`](https://huggingface.co/datasets/theatticusproject/cuad-qa).
Each train/test example is a SQuAD-style record:
`(title, context, question, answers={text, answer_start})`.

The retrieval stack we are evaluating:

| Stage | Component | Notes |
| --- | --- | --- |
| Chunking | done in `upload_to_qdrant.py` | Stored payload includes `char_start`, `char_end`, `page_start`, `page_end`, `title`, `doc_id`, `text`. |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-d | Trained on Reddit / S2ORC / MS MARCO / NLI / QA pairs; **CUAD not in training data** (per model card). |
| Retrieval | Qdrant `cuad_contracts`, cosine | `top_k` default 10. |
| Highlight rerank | `BAAI/bge-reranker-v2-m3` (XLM-R / BGE-M3 base) | Multilingual general reranker; **CUAD not listed in training data** (per model card). |
| RAG | `Qwen/Qwen3-235B-A22B:novita` via HF Inference | Out of scope for retrieval metrics; in scope only for source-faithfulness. |

None of the three models advertises CUAD in its training data
(see the model-card raw markdown at
<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/README.md>
and
<https://huggingface.co/BAAI/bge-reranker-v2-m3/raw/main/README.md>),
so a CUAD-derived gold set is, to a first approximation, a clean held-out
evaluation for the current production stack.

---

## 2. Ground truth: what counts as a "relevant" chunk?

### 2.1 Data source

The atomic unit of CUAD ground truth is an **annotated span**:
`(title, category, answer_text, answer_start, answer_end = answer_start + len(answer_text))`.

**Decision (user, 2026-05-28):** sole gold source is the HuggingFace dataset
repo **`theatticusproject/cuad`** — specifically `CUAD_v1/CUAD_v1.json`,
pulled via `huggingface_hub.hf_hub_download`. The sibling `cuad-qa` dataset
relies on a loading script that fetches from `github.com/TheAtticusProject/cuad`,
which the user explicitly disallows; the JSON content of the two is
equivalent. No GitHub fallback, no cross-check, no disagreement logging.

We will use the **HuggingFace `theatticusproject/cuad-qa`** train+test splits
combined, because:

- It is canonical and versioned. The original `master_clauses.csv` and the
  GitHub `CUADv1.json` are equivalent in content but harder to pin to a
  single revision. (We can fall back to the GitHub `CUADv1.json` if joining
  on `title` fails — see Open Question Q1.)
- Each row is keyed by `title`, which matches the `title` payload field
  already stored on every Qdrant point.

### 2.2 Mapping spans to Qdrant points

A Qdrant point `p` is relevant for query `Q` (where `Q` belongs to CUAD
category `c`) **iff** there exists an annotated span
`(title=p.title, category=c, answer_start, answer_end)` such that the
intervals `[p.char_start, p.char_end]` and `[answer_start, answer_end]`
overlap by **at least 1 character**.

**Why "≥ 1 char" not "≥ 50% of span"?** CUAD chunks (~600–1200 chars) are
usually larger than individual spans (the median CUAD answer is ~50–300
chars). Most relevant chunks contain the full span; the marginal case is a
chunk that straddles two CUAD spans, where any positive overlap is still
"yes, this chunk contains clause material the user wanted." A 50%-of-span
threshold would falsely penalize a correct retrieval whose chunk boundary
happened to bisect a long span. We will however **report span-coverage
statistics** in the report so we can revisit if reality looks different.

### 2.3 Empty-span (negative) categories

CUAD includes Yes/No categories where the annotated `text` is empty (the
contract does *not* contain that clause type). For those rows we treat the
contract as having **no relevant chunk** for that category, and any retrieval
result that returns a chunk from such a contract for the corresponding query
will contribute to false positives in Precision@k.

### 2.4 Ground-truth projections

The same underlying spans are projected into three artifacts at different
granularities. The **contract-level** projection (`gold_contracts.json`) is
the primary signal for cross-contract retrieval — "did `/search` surface a
contract that actually has this clause?" — and is robust to chunk-boundary
artifacts where the chunker's `char_start`/`char_end` may not align 1:1
with CUAD's `answer_start` coordinate system (the chunker indexes PDF-
extracted text, CUAD's spans index its own `context` string; they overlap
in the prefix but drift in long documents). The **chunk-level** projection
(`gold.json`) is the within-document highlight signal — "given a chunk we
retrieved, does it contain a CUAD-annotated span?" — and is intentionally
strict. The **span-level** projection (`gold_spans.json`) preserves the
raw annotated text + character offsets per (query, document) for a future
highlight-quality eval.

| File | Granularity | Key | Value |
| --- | --- | --- | --- |
| `gold.json` | Qdrant chunk | `qid` | `list[point_id]` overlapping any span of the query's category. |
| `gold_contracts.json` | Contract title | `qid` | `{category, form, relevant_titles, total_relevant}` — titles with ≥1 nonempty span of `category`, restricted to titles ingested in Qdrant. |
| `gold_spans.json` | Raw CUAD span | `qid` | `{category, form, spans_by_document: {title: [{answer_text (≤300 chars), char_start, char_end}, ...]}}` — for future highlight-hit eval, unused by `run_eval.py` today. |

`run_eval.py` consumes the first two and reports `chunk_metrics` and
`contract_metrics` side by side. Contract metrics treat the rank-ordered
top-k titles (with duplicates preserved at their original ranks) as the
retrieved list; recall denominator is `total_relevant`.

---

## 3. Query construction

CUAD ships **two** natural query forms per category:

1. **Canonical clause name** (short, e.g. `"Change Of Control"`).
2. **Question prompt** (long, e.g. `"Highlight the parts (if any) of this
   contract related to 'Change of Control' that should be reviewed by a
   lawyer. Details: ..."`) — shipped in the `question` field of every
   CUAD-QA row.

**Recommendation: use BOTH, tagged separately, with the canonical clause
name as the primary headline metric.**

Rationale:
- The short form ("Governing Law", "Indemnification") is what an actual user
  of the demo UI types. Optimizing on the long question form risks making
  the system look better than it is in production.
- The long question form is what an LLM-mediated chat client would emit,
  and is a useful generalization signal (does the embedder handle the
  paraphrase?).
- Recording both at once is nearly free and prevents a future tuning round
  from silently changing which form we measure against.

Concretely, `queries.json` will have entries shaped like:

```json
{ "qid": "governing_law__short",    "category": "Governing Law", "q": "Governing Law", "form": "short" }
{ "qid": "governing_law__question", "category": "Governing Law", "q": "Highlight the parts (if any) ...", "form": "question" }
```

41 categories × 2 forms = **82 corpus-wide queries**. Document-scoped queries
(see §4) will reuse the same 82 query texts but be repeated per sampled
document, so the total query count is `82 × (1 + |doc_sample|)`.

The eight legacy queries already in this file's spec
("indemnification clause", "termination for convenience", …) are kept
verbatim in `queries.json` under `form: "legacy"` so the small smoke-test
loop still works. New, never substituted.

---

## 4. Partitions

### 4.1 Corpus-wide retrieval (primary)

For each `(category, form)` query, call `/search?q=…&top_k=20` with no
document filter. The gold set is the **union over all 510 contracts** of
every Qdrant point whose `(title, char_start, char_end)` overlaps any
annotated span of `category`.

This measures: "given a clause category, can the system find all relevant
passages anywhere in the corpus?"

### 4.2 Document-scoped retrieval (secondary)

Sample **30 contracts** (stratified across the 25 CUAD contract types so
each type has ≥1 representative). For each sampled contract and each
`(category, form)` query, call
`/search?q=…&document_name=<title>&top_k=20`. The gold set is restricted to
that one document.

This measures: "within a single contract, does ranking surface the right
clause first?" — closer to the legal-review use case, and the regime where
MRR matters most because there is usually exactly one (or a small number of)
relevant chunks.

30 was picked to keep eval runtime tractable
(30 docs × 82 queries × 1 request ≈ 2,460 requests) while still being
large enough for a reasonable mean. **Reproducibility:** we seed the sample
with `random.Random(42).sample(titles_sorted, 30)` so the partition is
stable across runs.

### 4.3 Held-out partition for future-proofing

Hold out **50 contracts** (stratified, seed=`1337`) that are **excluded
from every metric reported as the headline number**. Today this matters
little because the current models are zero-shot. It matters the moment
anyone evaluates a CUAD-fine-tuned model (see §6). The held-out set's
results are still computed and stored, just under a separate
`held_out_*` key in the run JSON.

---

## 5. Metrics

All metrics are computed per `(category, form)` query and then aggregated
two ways: macro-averaged across categories (headline) and per-category
(diagnostic table).

### 5.1 Retrieval quality

Let `R_k(Q)` be the top-k Qdrant point IDs returned for query `Q`, and
`G(Q)` the gold set.

- **Recall@k**, k ∈ {5, 10, 20}:
  `|R_k(Q) ∩ G(Q)| / |G(Q)|`. Skip the query if `|G(Q)| = 0`.
- **Precision@k**, k ∈ {5, 10}:
  `|R_k(Q) ∩ G(Q)| / k`.
- **MRR@20**:
  `1 / rank_of_first_relevant`, 0 if none in top 20.
- **nDCG@10**:
  Binary relevance for the headline number: `rel_i = 1 if R_k[i] ∈ G(Q) else 0`,
  `DCG = Σ rel_i / log2(i + 2)`, normalized by ideal DCG over `min(|G(Q)|, 10)`
  relevant items.

**Why binary?** CUAD annotations are intrinsically binary per category:
a span is labeled or it is not. Inventing a graded scheme
("sibling category = 1") would require an externally defined category
ontology that CUAD does not ship; doing it ourselves would introduce
researcher bias into the headline metric. We will compute a *secondary*
graded variant later if §8 Open Q3 says yes — but only as a diagnostic,
not the headline.

### 5.2 Latency

For every query, also record:
- Wall-clock per request (5 repetitions, p50 and p95).
- Per-stage timing only if surfaced by the API
  (currently it isn't — see `app.py`; out of scope to add).

### 5.3 Highlight quality (in-chunk)

The highlight reranker scores sentences inside a returned chunk. For each
chunk that is in `G(Q)`, we check whether the **top-ranked highlighted
sentence** is itself part of an annotated CUAD span (any 1-char overlap with
the chunk-local slice of the span). Aggregate as
**Highlight-Hit@1 across all gold chunks**. This is the only way to put a
deterministic number on "is the highlight pointing at the right sentence."

### 5.4 `/chat` faithfulness (deterministic, no LLM judge)

For each `(category, form)` query, hit `POST /chat`. The response includes
`sources` (each with a `title`). The check:

- **Source-Title Coverage**: at least one cited `source.title` is a contract
  that has ≥ 1 annotated span of `category`. Boolean per query; report
  fraction of queries that pass.

We deliberately do **not** compute answer-text similarity to gold spans
with an LLM judge. The user explicitly disallowed judge-model evals, and
source-title coverage is sufficient to catch the regressions we care about
("chat started citing the wrong contract" or "chat fell back to
'I cannot determine'"). Answer length and per-query character count are
recorded as soft signals only.

### 5.5 Regression thresholds (for `report.md` headline)

These match the operating contract in the agent's system prompt:

| Metric | Regression iff |
| --- | --- |
| Latency p95 (any query) | > 20% increase vs baseline |
| Recall@10 (macro) | absolute drop ≥ 2 pp |
| MRR@20 (macro) | absolute drop ≥ 2 pp |
| nDCG@10 (macro) | absolute drop ≥ 2 pp |
| Top-5 highlight set per query | > 2 of 5 sentences changed (unless flagged) |
| `/chat` source-title coverage | any drop |
| `/chat` answer is "I cannot determine…" where baseline had real answer | hard regression, surface immediately |
| Any query returning 0 results when baseline returned ≥ 1 | hard regression, surface immediately |

---

## 6. Overfitting and leakage

CUAD is a popular public benchmark. The risk is that a future agent swaps
in a model that was trained on CUAD labels, gets inflated numbers on our
eval set, and ships a regression for real users.

### 6.1 Current production stack — leakage check

| Model | CUAD in training data? | Evidence |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | **No** | Training-data table lists 32 sources, all general web/QA/NLI (Reddit, S2ORC, MS MARCO, NLI, …). No legal-contract dataset. (<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/README.md>) |
| `BAAI/bge-reranker-v2-m3` | **No (per public model card)** | Card describes the model as a multilingual general-purpose reranker derived from BGE-M3 (XLM-R-large). No legal corpus is named. Treating as low-risk. (<https://huggingface.co/BAAI/bge-reranker-v2-m3/raw/main/README.md>) |
| `Qwen/Qwen3-235B-A22B` | Unknown; likely web-scale pretraining that includes the public CUAD corpus | Large general LLM. We are not benchmarking the model's parametric memory, only its faithfulness to retrieved sources, so this is acceptable. |

Conclusion: a CUAD-derived gold set is, today, a **legitimate held-out
test** for the embedder and the reranker.

### 6.2 Hard rule: models that ARE trained on CUAD

The CUAD paper releases fine-tuned **RoBERTa-base / RoBERTa-large /
DeBERTa-v2-xlarge** checkpoints trained directly on the CUAD QA labels
(<https://github.com/TheAtticusProject/cuad>). There are also community
fine-tunes like `akdeniz27/roberta-large-cuad`.

**Rule:** these checkpoints MUST NOT replace the production retriever or
reranker on the basis of this eval, because their numbers on this eval are
inflated by direct label memorization. If anyone does run one for an
upper-bound sanity ceiling, the result must be filed in `report.md`
under a clearly marked **"leakage-aware ceiling"** section, never as the
headline number, and the run directory must be named
`runs/<ts>__LEAKAGE/` so it is unmistakable.

### 6.3 Other anti-leakage hygiene

- The held-out 50-contract partition (§4.3) is preserved indefinitely.
  Any future fine-tuned model gets evaluated **only** on the held-out set
  for its headline numbers.
- `queries.json` is append-only. Removing or rewording an existing query
  would invalidate the baseline.
- Do not use the GPT-4-augmented question rephrasings that some CUAD
  derivatives publish; stay with the original question prompts shipped in
  `cuad-qa`.
- Do not train chunking or reranker hyperparameters against this eval set.
  If we ever do, a **second** held-out set is required.

---

## 7. Build plan (ordered, each a single follow-up task)

1. **Pull CUAD annotations.** Download `theatticusproject/cuad-qa` (train+test
   merged) via `datasets.load_dataset` and write a normalized JSONL to
   `tests/eval/cuad_gold/annotations.jsonl` with fields
   `{title, category, answer_text, answer_start, answer_end}`. Categories
   come from parsing the `question` field's quoted category name
   (`"Highlight the parts (if any) of this contract related to '<X>' ..."`).
2. **Generate `tests/eval/queries.json`** from the 41 categories × 2 forms,
   plus the 8 legacy smoke queries from the agent system prompt. Stable
   `qid` for every entry.
3. **Write `tests/eval/build_gold.py`.** Scrolls all Qdrant points, joins
   to `annotations.jsonl` on `title`, computes character-interval overlap,
   emits `tests/eval/cuad_gold/gold.json` keyed by `qid → list[point_id]`
   (and for doc-scoped, `(qid, title) → list[point_id]`). Also emits a
   `gold_stats.json`: |G(Q)| per query, coverage histograms.
4. **Write `tests/eval/run_eval.py`.** < 200 lines. Loads `quer ies.json`,
   hits `/search` (5x for latency, 1x for ranking, top_k=20) and `/chat`
   (1x). Dumps per-query JSON under `tests/eval/runs/<YYYY-MM-DD-HHMM>/`.
   Computes Recall@{5,10,20} / Precision@{5,10} / MRR@20 / nDCG@10 /
   Highlight-Hit@1 / source-title-coverage / latency p50/p95.
5. **First baseline run.** App must already be up; agent confirms then runs
   `run_eval.py` and copies the run to `tests/eval/baseline/`. Commit
   `baseline/`, `queries.json`, `build_gold.py`, `run_eval.py`,
   `STRATEGY.md`. Add `tests/eval/runs/` and `tests/eval/cuad_gold/`
   to `.gitignore` (cuad_gold is large and re-derivable).
6. **Write `tests/eval/report.md` template** with one-line headline and a
   numeric diff table; `run_eval.py` regenerates it on every run.
7. **(Optional, later)** Add a `--leakage-ceiling` mode that swaps in
   `akdeniz27/roberta-large-cuad` or similar purely as a sanity ceiling,
   stored under `runs/<ts>__LEAKAGE/`. Off by default.

Estimated runtime for one full eval pass (corpus-wide 82 queries +
30 doc-scoped × 82 = 2,542 search calls + 82 chat calls, 5x latency reps
amortized to 2x for ranking, 1x for full snapshot): **dominated by the
current ~20 s /search p95**, so **~14 hours** with sequential calls.
This is fine for an overnight baseline. After `search-perf` lands its
async highlight changes the same pass should drop to **<10 min**.

---

## 8. Anti-foot-guns (operating rules for this agent)

- Will NOT edit `app.py`, `qdrant_search_hf.py`, `chat_hf.py`, or any
  ingestion code. Performance fixes are `search-perf`'s lane; ingestion
  changes are `cuad-ingest`'s lane.
- Will NOT remove, reword, or "fix" entries in `queries.json` after first
  commit. New queries are appended.
- Will NOT compute any "score" with an LLM-as-judge for headline numbers.
- Will NOT commit `tests/eval/runs/` or `tests/eval/cuad_gold/`.
- Will NOT read `.env*` files. Config comes from `qdrant_cluster_connect.py`
  and `chat_hf.py`.
- Will report and stop if the API is not up on port 8000; will not start
  uvicorn unless asked explicitly.
- A query that drops from "≥1 result" to "0 results" against baseline, or
  a `/chat` answer that flips to "I cannot determine…", is surfaced
  immediately at the top of `report.md`, not buried in the table.

---

## 9. Open questions for the user

1. **Source of truth for CUAD annotations:** HuggingFace
   `theatticusproject/cuad-qa` (recommended; canonical, versioned) vs the
   original GitHub `CUADv1.json` / `master_clauses.csv`? They are
   equivalent in content but the HF version is easier to pin.
2. **Document-scoped sample size:** 30 contracts (proposed) feels right
   for an overnight run. Bump to 50 once `/search` p95 drops below 1 s?
   Or keep at 30 forever for comparability?
3. **Graded relevance:** should we ship a secondary graded-nDCG variant
   alongside binary? If yes we need a category-sibling map (e.g.
   *Non-Compete* ↔ *Exclusivity*), which is a judgment call. Default
   answer: no, binary only.
4. **Leakage-ceiling baseline:** do you want a one-off run with a
   CUAD-fine-tuned model (e.g. `akdeniz27/roberta-large-cuad`) as a
   labeled "ceiling" number, or skip it entirely? Default: skip until
   asked.
5. **Reuse of legacy 8-query smoke set:** keep it as `form: "legacy"`
   inside the main `queries.json` (simpler), or split into a separate
   `smoke_queries.json` (cleaner)? Default: keep in one file with a
   `form` discriminator.
