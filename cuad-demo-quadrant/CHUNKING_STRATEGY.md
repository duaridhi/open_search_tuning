# Text Chunking Strategy for CUAD Contract Embeddings

## Overview

The CUAD PDF ingestion pipeline uses a **smart hierarchical boundary-aware chunking strategy** to split contract text into semantically meaningful chunks before embedding and uploading to Qdrant vector database.

## Chunking Strategy

### Core Algorithm: `split_text_with_offsets()`

Located in `upload_to_qdrant.py`, the chunking function implements a **sliding window with intelligent boundary detection**:

```python
def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    separators = ["\n\n", "\n", " ", ""]
    # ... splits at best available boundary
```

### Strategy Details

#### 1. **Hierarchical Boundary Detection**
The algorithm attempts to break text at natural boundaries in this priority order:
- **Paragraph boundaries** (`"\n\n"`) — highest priority
- **Line boundaries** (`"\n"`)
- **Word boundaries** (` ` space)
- **Character boundaries** (`""`) — fallback only

This ensures chunks respect document structure and semantic coherence.

#### 2. **How It Works**

```
┌─────────────────────────────────────────────────────────┐
│ Full Contract Text                                       │
└─────────────────────────────────────────────────────────┘
         ↓
    [Start at offset 0]
         ↓
┌─────────────────────────────────────────────────────────┐
│ Try to fit chunk_size (500 chars) of text               │
│ Then search backwards for nearest separator             │
│       [next_chunk_boundary] ← found at \n\n            │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Yield Chunk 1: { text, char_start, char_end }           │
└─────────────────────────────────────────────────────────┘
         ↓
    [Advance by (end - overlap)]
    [Move to next chunk with 50-char overlap]
         ↓
    [Repeat until text exhausted]
```

#### 3. **Overlap for Context Preservation**

- Default overlap: **50 characters**
- Ensures semantic continuity between chunks
- Important for vector search — similar content appears in adjacent chunks
- Example:
  ```
  Chunk 1 ends: "...payment shall be made within 30 days."
  Chunk 2 starts: "...within 30 days. Any late payment will incur..."
                     ↑ overlapping context from Chunk 1
  ```

### Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CHUNK_SIZE` | 500 chars | Target size per chunk |
| `CHUNK_OVERLAP` | 50 chars | Context overlap between chunks |
| `MAX_DOCS` | 1000 chunks | Total chunks to upload |
| `ENCODE_BATCH_SIZE` | 32 | Chunks per embedding batch |
| `UPLOAD_BATCH_SIZE` | 100 | Chunks per Qdrant upsert |

## Rationale for This Strategy

### 1. **Semantic Preservation** 
- Contracts are highly structured documents with clear logical boundaries (clauses, sections, subsections)
- Breaking at paragraph/line boundaries keeps related sentences together
- Avoids splitting mid-sentence or mid-clause, which would lose meaning

### 2. **RAG Optimization**
- For retrieval-augmented generation (RAG), chunks should be semantically cohesive
- When a chunk is retrieved as context, the model expects complete thoughts
- Boundary-aware chunking improves relevance of retrieved context

### 3. **Legal Document Specificity**
- Contract clauses often span multiple lines
- Paragraph breaks typically separate distinct legal obligations or definitions
- Natural language models (embedding models) perform better on semantically complete units

### 4. **Overlap for Robustness**
- Query might match content spanning two chunks
- Overlap ensures retrieval captures complete relevant context
- Trade-off: 50 chars is minimal overhead (10% of 500), benefits retrieval accuracy significantly

### 5. **Fallback Handling**
- If a 500-char window has no natural boundaries within the middle 50% (chunk_size // 2 = 250 chars), the algorithm falls back to character-level splitting
- This prevents infinite loops on text without natural boundaries (e.g., URLs, base64, code blocks)
- Rare in contract text but handles edge cases gracefully

## Implementation Flow

```
┌─ Extract Text from PDF (all pages) ─────────────────┐
│                                                      │
├─ Concatenate pages with \n\n separator              │
│                                                      │
├─ Build page_map: map character offsets → page #     │
│                                                      │
├─ For each PDF:                                       │
│   ├─ Split full text into chunks using boundaries   │
│   │                                                  │
│   └─ For each chunk:                                │
│       ├─ Calculate which pages it spans             │
│       ├─ Store metadata:                            │
│       │  - doc_id, title, text                      │
│       │  - char_start/end (for PDF highlighting)    │
│       │  - page_start/end (for citation)            │
│       │  - pdf_path (for traceability)              │
│       │                                              │
│       └─ Yield for embedding                        │
│                                                      │
└─ Batch encode embeddings & upsert to Qdrant ────────┘
```

## Metadata Attached to Each Chunk

```python
{
    "doc_id": "contract-name-chunk-42",      # Unique ID
    "title": "contract-name",                # PDF filename stem
    "text": "...",                           # Actual text (500 chars)
    "char_start": 1250,                      # Start offset in full text
    "char_end": 1750,                        # End offset in full text
    "page_start": 3,                         # Page(s) this chunk spans
    "page_end": 4,
    "pdf_path": "path/to/contract.pdf",     # Relative path for retrieval
}
```

This metadata enables:
- **Exact citation**: Link search results directly to PDF pages
- **Traceability**: Know which contract and location each embedding came from
- **Transparency**: Show users the source text chunk alongside similarity scores

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Chunk size | ~500 chars | ≈ 150–200 tokens for language models |
| Overlap | 50 chars | ≈ 15–20 tokens |
| Effective unique content per chunk | ~450 chars | (500 - 50 overlap) |
| Max chunks per contract | Varies | Depends on contract length |
| Processing batches | 100-chunk upserts | Balances memory vs. network round-trips |

## Why NOT Other Strategies?

### ❌ Fixed Character-only Splitting
- Would break contracts mid-clause
- "...party shall" | "...indemnify" (semantically broken)
- Poor retrieval quality

### ❌ Sentence-based Splitting
- Contract "sentences" can be very long (100+ tokens)
- Would create inconsistent chunk sizes (too large for some)
- "Sentence" tokenization unreliable in legal text (e.g., "U.S." is not end of sentence)

### ❌ No Overlap
- Query might start at the end of one chunk and continue into next
- Without overlap, key context would be missing from retrieved chunk
- Significantly degrades RAG quality

### ✅ This Strategy: Hierarchical Boundaries with Overlap
- Respects document structure (legal precision)
- Consistent encoding/retrieval (same chunk sizes across collection)
- Robust against edge cases (fallback to character level)
- Context-aware retrieval (overlap ensures complete thoughts)

## Usage Example

```bash
# Set environment variables
export QDRANT_API_KEY="your_api_key"
export CLUSTER_URL="https://your-qdrant-cluster"
export MAX_DOCS=5000          # Upload 5000 chunks (instead of default 1000)
export CHUNK_SIZE=800         # Larger chunks (instead of default 500)
export CHUNK_OVERLAP=100      # More overlap (instead of default 50)

# Run ingestion
python upload_to_qdrant.py
```

## Monitoring & Tuning

When running uploads, watch for:

```
[DEBUG] {title}: {N} pages, {M:,} chars
```

**If chunks seem too large:**
- Reduce `CHUNK_SIZE` (e.g., 300–400) for more granular retrieval
- Increases total chunks and computation time

**If chunks seem too small:**
- Increase `CHUNK_SIZE` (e.g., 700–1000) for broader context
- Reduces total chunks but may lose fine-grained retrieval

**If retrieval is imprecise:**
- Increase `CHUNK_OVERLAP` to 100–150 for richer context overlap
- Improves recall at small storage cost

## Summary

This chunking strategy balances:
- **Semantic coherence** (hierarchical boundaries)
- **Retrieval accuracy** (overlap for context)
- **Practical efficiency** (reasonable batch sizes)
- **Legal precision** (metadata for exact citations)

The result is chunks that embeddings models can meaningfully represent, and that users can trace back to exact contract locations.
