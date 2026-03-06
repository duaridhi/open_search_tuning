"""
test_split_offsets.py
---------------------
Standalone tests for split_text_with_offsets().

Verifies:
 1. Boundary correctness — text[char_start:char_end] == chunk["text"]
 2. Overlap correctness  — last `chunk_overlap` chars of chunk N
                           == first `chunk_overlap` chars of chunk N+1

No OpenSearch / model dependencies; safe to run at any time.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline the splitter so this file has zero external dependencies
# ---------------------------------------------------------------------------
def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """Split text into chunks of ~chunk_size chars with overlap.
    Returns a list of dicts with keys: text, char_start, char_end.
    Splits preferentially on paragraph -> newline -> space boundaries.
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if end < len(text):
            split_at = -1
            for sep in separators:
                idx = chunk_text.rfind(sep)
                if idx > chunk_size // 2:
                    split_at = idx + len(sep)
                    break
            if split_at > 0:
                end = start + split_at
                chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "char_start": start, "char_end": end})
        if end >= len(text):   # reached the end — stop; avoids tail micro-chunk loop
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


# Production settings (keep in sync with 02_index_cuad_documents.py)
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

CUAD_JSON_PATH = (
    Path(__file__).resolve().parent.parent / "cuad_data" / "CUAD_v1" / "CUAD_v1.json"
)


# ---------------------------------------------------------------------------
# Core test harness
# ---------------------------------------------------------------------------
def test_split_boundaries(
    text: str, chunk_size: int = 100, chunk_overlap: int = 20, label: str = ""
) -> bool:
    """
    Split *text*, print every chunk's boundaries and content, and verify:
      1. char_start / char_end slice the original text correctly.
      2. The overlap window is preserved between consecutive chunks.

    Returns True if all assertions pass.
    """
    chunks = split_text_with_offsets(text, chunk_size, chunk_overlap)
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*70}")
    print(f"TEST{tag}  chunk_size={chunk_size}  chunk_overlap={chunk_overlap}")
    print(f"  Input length : {len(text)} chars")
    print(f"  Chunks found : {len(chunks)}")
    print(f"{'='*70}")

    boundary_errors: list[int] = []
    overlap_errors:  list[int] = []

    for i, chunk in enumerate(chunks):
        s, e = chunk["char_start"], chunk["char_end"]
        actual_text = text[s:e]

        # 1. Boundary check
        boundary_ok = actual_text == chunk["text"]
        if not boundary_ok:
            boundary_errors.append(i)

        # 2. Overlap check against previous chunk
        overlap_snippet = ""
        overlap_ok = True
        if i > 0:
            prev      = chunks[i - 1]
            prev_text = text[prev["char_start"]:prev["char_end"]]
            if len(prev_text) >= chunk_overlap and len(actual_text) >= chunk_overlap:
                expected_overlap = prev_text[-chunk_overlap:]
                actual_head      = actual_text[:chunk_overlap]
                overlap_ok       = expected_overlap == actual_head
                overlap_snippet  = repr(actual_head)
                if not overlap_ok:
                    overlap_errors.append(i)

        # Print row
        boundary_flag = "" if boundary_ok else "  !! BOUNDARY MISMATCH"
        overlap_flag  = "" if overlap_ok  else "  !! OVERLAP MISMATCH"
        preview = repr(actual_text[:60]) + ("..." if len(actual_text) > 60 else "")
        print(
            f"  Chunk {i:>3}  [{s:>6} – {e:>6}]  len={e - s:>4}  {preview}"
            + boundary_flag
        )
        if i > 0:
            if overlap_snippet:
                print(f"           overlap head : {overlap_snippet}{overlap_flag}")
            else:
                print(f"           overlap       : (chunk shorter than overlap window — skipped)")

    # Summary
    print(
        f"\n  Boundary errors : {len(boundary_errors)}"
        + (f"  (chunks {boundary_errors})" if boundary_errors else "  ✓")
    )
    print(
        f"  Overlap errors  : {len(overlap_errors)}"
        + (f"  (chunks {overlap_errors})" if overlap_errors else "  ✓")
    )
    passed = not boundary_errors and not overlap_errors
    print(f"  Result          : {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_all_tests():
    results: dict[str, bool] = {}

    # ── Test 1: short controlled text (easy to trace by eye) ────────────
    short_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
        "The five boxing wizards jump quickly."
    )
    results["short text"] = test_split_boundaries(
        short_text, chunk_size=60, chunk_overlap=15, label="short text"
    )

    # ── Test 2: multi-paragraph contract excerpt (exercises \n\n splits) ─
    multi_para = (
        "This Agreement is entered into as of January 1, 2024.\n\n"
        "WHEREAS, the parties wish to set forth their agreement in writing.\n\n"
        "NOW, THEREFORE, in consideration of the mutual covenants contained herein, "
        "the parties agree as follows:\n\n"
        "1. Term. The term of this Agreement shall commence on the Effective Date "
        "and shall continue for a period of two (2) years unless earlier terminated.\n\n"
        "2. Services. Contractor shall provide the services described in Exhibit A "
        "attached hereto and incorporated herein by reference.\n\n"
        "3. Payment. Client shall pay Contractor a monthly fee of $10,000, due on "
        "the first business day of each calendar month during the Term."
    )
    results["contract excerpt"] = test_split_boundaries(
        multi_para,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        label="contract excerpt",
    )

    # ── Test 3: edge cases ───────────────────────────────────────────────
    # 3a. Text shorter than one chunk
    results["text < chunk_size"] = test_split_boundaries(
        "Short.", chunk_size=50, chunk_overlap=10, label="text < chunk_size"
    )
    # 3b. Text exactly chunk_size chars
    exact = "x" * CHUNK_SIZE
    results["text == chunk_size"] = test_split_boundaries(
        exact, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, label="text == chunk_size"
    )
    # 3c. No natural separator (forces char-level split)
    no_sep = "a" * (CHUNK_SIZE * 3)
    results["no separators"] = test_split_boundaries(
        no_sep, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, label="no separators"
    )

    # ── Test 4: first real CUAD contract (production settings) ───────────
    if CUAD_JSON_PATH.exists():
        with open(CUAD_JSON_PATH) as f:
            cuad_contracts = json.load(f)["data"]
        real_context = cuad_contracts[0]["paragraphs"][0]["context"]
        results["CUAD contract 0"] = test_split_boundaries(
            real_context,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            label="CUAD contract 0",
        )
    else:
        print(f"\n[SKIP] CUAD JSON not found at {CUAD_JSON_PATH}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False
    print(f"{'='*70}")
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*70}\n")
    return all_passed


if __name__ == "__main__":
    run_all_tests()
