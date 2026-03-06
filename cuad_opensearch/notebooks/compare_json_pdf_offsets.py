"""
compare_json_pdf_offsets.py
────────────────────────────
Compares the text extracted from CUAD_v1.json against text extracted from the
corresponding PDF file.  Then builds a character-offset mapping so that JSON
answer_start positions can be translated to PDF char offsets.

Usage
─────
    python compare_json_pdf_offsets.py
    # or for a different contract title substring:
    TITLE=ENDORSEMENT python compare_json_pdf_offsets.py

Algorithm
─────────
Word-level SequenceMatcher (much faster than char-level):
  1. Tokenise both texts into (word, char_offset) pairs.
  2. Find matching word-sequences between JSON tokens and PDF tokens.
  3. Each matched pair of words becomes an (anchor_json_off, anchor_pdf_off) point.
  4. For any json_off in between two anchors, interpolate to get pdf_off.
  5. Snap the result by searching for the target text in a ±window in the PDF.
"""

import bisect
import json
import os
import re
import sys
from pathlib import Path

import pdfplumber

# ── Configuration ────────────────────────────────────────────────────────────
CUAD_JSON  = Path(__file__).resolve().parents[1] / "cuad_data" / "CUAD_v1" / "CUAD_v1.json"
PDF_ROOT   = Path(__file__).resolve().parents[1] / "cuad_data" / "CUAD_v1" / "full_contract_pdf"
TITLE_HINT = os.getenv("TITLE", "LEGACYTECHNOLOGY")
SNAP_WINDOW = 200   # chars either side of the interpolated position to search for an exact match


# ── Helpers ──────────────────────────────────────────────────────────────────
def tokenise(text: str) -> list[tuple[str, int]]:
    """Return list of (token, char_start) for every non-whitespace run."""
    return [(m.group(), m.start()) for m in re.finditer(r'\S+', text)]


def build_anchors(json_text: str, pdf_text: str):
    """
    Word-level SequenceMatcher alignment.
    Returns (json_anchors, pdf_anchors) — parallel sorted int lists of
    char offsets where the two strings provably agree.
    """
    import difflib

    j_toks = tokenise(json_text)
    p_toks = tokenise(pdf_text)
    j_words = [t[0] for t in j_toks]
    p_words = [t[0] for t in p_toks]

    sm = difflib.SequenceMatcher(None, j_words, p_words, autojunk=True)
    blocks = sm.get_matching_blocks()

    j_anchors = [0]
    p_anchors = [0]
    for blk in blocks:
        if blk.size == 0:
            continue
        for k in range(blk.size):
            j_tok = j_toks[blk.a + k]
            p_tok = p_toks[blk.b + k]
            j_anchors.append(j_tok[1])
            p_anchors.append(p_tok[1])
            j_anchors.append(j_tok[1] + len(j_tok[0]))
            p_anchors.append(p_tok[1] + len(p_tok[0]))

    j_anchors.append(len(json_text))
    p_anchors.append(len(pdf_text))
    return j_anchors, p_anchors


def interpolate(json_off: int, j_anchors: list[int], p_anchors: list[int]) -> int:
    """Piecewise-linear interpolation: json char offset → pdf char offset."""
    idx = bisect.bisect_right(j_anchors, json_off) - 1
    idx = max(0, min(idx, len(j_anchors) - 2))
    j0, j1 = j_anchors[idx], j_anchors[idx + 1]
    p0, p1 = p_anchors[idx], p_anchors[idx + 1]
    if j1 == j0:
        return p0
    frac = (json_off - j0) / (j1 - j0)
    return int(p0 + frac * (p1 - p0))


def _ws_norm(s: str) -> str:
    """Collapse any run of whitespace (including non-breaking spaces) to a single space."""
    return re.sub(r'[\s\u00a0]+', ' ', s).strip()


def map_json_to_pdf(
    json_off: int,
    json_text_len: int,
    j_anchors: list[int],
    p_anchors: list[int],
    pdf_text: str,
    answer_text: str,
    snap_window: int = SNAP_WINDOW,
) -> tuple[int, bool]:
    """
    Map a JSON char offset to a PDF char offset.

    1. Interpolate to get a candidate position.
    2. In pdf_text[candidate-window : candidate+window]:
         a. Try exact substring search.
         b. Try case-insensitive search.
         c. Try whitespace-normalised search (handles double-space padding in JSON).
    3. If any search succeeds, return the snapped position (snapped=True).
    4. Otherwise return the interpolated position (snapped=False).

    Returns (pdf_char_offset, snapped_exactly).
    """
    candidate = interpolate(json_off, j_anchors, p_anchors)
    lo = max(0, candidate - snap_window)
    hi = min(len(pdf_text), candidate + snap_window + len(answer_text))
    window = pdf_text[lo:hi]

    # a. Exact
    idx = window.find(answer_text)
    if idx >= 0:
        return lo + idx, True

    # b. Case-insensitive
    idx = window.lower().find(answer_text.lower())
    if idx >= 0:
        return lo + idx, True

    # c. Whitespace-normalised: build a token-start map of the window so we can
    #    find where the normalised match begins in the original window.
    norm_answer  = _ws_norm(answer_text)
    norm_window  = _ws_norm(window)
    idx_norm = norm_window.find(norm_answer)
    if idx_norm < 0:
        idx_norm = norm_window.lower().find(norm_answer.lower())
    if idx_norm >= 0:
        # Map normalised index back to the original window index via token alignment.
        tokens = list(re.finditer(r'\S+', window))
        norm_tokens = list(re.finditer(r'\S+', norm_window))
        # Find which norm_token the norm match starts at.
        tok_idx = bisect.bisect_right([t.start() for t in norm_tokens], idx_norm) - 1
        tok_idx = max(0, min(tok_idx, len(tokens) - 1))
        return lo + tokens[tok_idx].start(), True

    return candidate, False


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load JSON ──────────────────────────────────────────────────────────────
    print(f"Loading CUAD JSON …")
    data = json.loads(CUAD_JSON.read_text())
    entry = next((i for i in data["data"] if TITLE_HINT.upper() in i["title"].upper()), None)
    if entry is None:
        sys.exit(f"No contract found matching title hint '{TITLE_HINT}'")

    title = entry["title"]
    json_ctx = entry["paragraphs"][0]["context"]
    qas = entry["paragraphs"][0]["qas"]
    answered = [q for q in qas if q["answers"]]

    print(f"Title         : {title}")
    print(f"JSON context  : {len(json_ctx):,} chars   answered Q&A: {len(answered)}")

    # ── Find matching PDF ──────────────────────────────────────────────────────
    # PDF stem may differ slightly in casing; search case-insensitively
    candidates = [p for p in PDF_ROOT.rglob("*") if p.suffix.upper() == ".PDF"
                  and title.upper() in p.stem.upper()]
    if not candidates:
        sys.exit(f"No PDF found for title '{title}'")
    pdf_path = candidates[0]
    print(f"PDF source    : {pdf_path.relative_to(PDF_ROOT)}")

    # ── Extract PDF text ───────────────────────────────────────────────────────
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pg in pdf.pages:
            pages.append(pg.extract_text() or "")
    pdf_text = "\n\n".join(pages)
    print(f"PDF text      : {len(pdf_text):,} chars   pages: {len(pages)}")
    print(f"PDF/JSON ratio: {len(pdf_text)/len(json_ctx):.3f}")
    print()

    # ── Head comparison ────────────────────────────────────────────────────────
    print("── First 300 chars comparison ─────────────────────────────────────")
    print(f"[JSON] {repr(json_ctx[:150])}")
    print(f"[PDF ] {repr(pdf_text[:150])}")
    print()

    # ── Build alignment ────────────────────────────────────────────────────────
    print("Building word-level alignment …", end=" ", flush=True)
    j_anchors, p_anchors = build_anchors(json_ctx, pdf_text)
    n_anchors = (len(j_anchors) - 2) // 2  # exclude padding endpoints
    matched_json_chars = sum(
        j_anchors[i + 1] - j_anchors[i]
        for i in range(0, len(j_anchors) - 1, 2)
        if j_anchors[i] != j_anchors[i + 1]
    )
    print(f"done.  {n_anchors:,} anchor pairs")
    print()

    # ── Map Q&A answers ────────────────────────────────────────────────────────
    print(f"── Q&A answer offset mapping ({min(len(answered),15)} of {len(answered)}) ───────────────────────")
    hdr = f"  {'':1} {'json_off':>9}  {'pdf_off':>9}  {'json_text':<32}  {'pdf_text':<32}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    exact = snapped = miss = 0
    for q in answered[:15]:
        a = q["answers"][0]
        j_off = a["answer_start"]
        j_text = a["text"]

        pdf_off, snapped_ok = map_json_to_pdf(
            j_off, len(json_ctx), j_anchors, p_anchors, pdf_text, j_text
        )
        p_text = pdf_text[pdf_off: pdf_off + len(j_text)]

        if _ws_norm(j_text) == _ws_norm(p_text):
            mark = "✓"; exact += 1
        elif snapped_ok:
            mark = "~"; snapped += 1
        else:
            mark = "✗"; miss += 1

        print(
            f"  {mark} {j_off:>9}  {pdf_off:>9}  "
            f"{repr(j_text[:30]):<32}  {repr(p_text[:30]):<32}"
        )

    print(f"\n  exact={exact}  near-match={snapped}  miss={miss}  / {min(len(answered),15)}")
    print()

    # ── Chunk-level demonstration ──────────────────────────────────────────────
    print("── Mapping Q&A offsets → PDF chunk index ────────────────────────────")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    def split_text_with_offsets(text, chunk_size=500, chunk_overlap=50):
        separators = ["\n\n", "\n", " ", ""]
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if end < len(text):
                for sep in separators:
                    idx = chunk_text.rfind(sep)
                    if idx > chunk_size // 2:
                        end = start + idx + len(sep)
                        chunk_text = text[start:end]
                        break
            chunks.append({"text": chunk_text, "char_start": start, "char_end": end})
            if end >= len(text):
                break
            ns = end - chunk_overlap
            if ns <= start:
                ns = start + 1
            start = ns
        return chunks

    pdf_chunks = split_text_with_offsets(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)

    def find_chunk(pdf_off, chunks):
        for i, c in enumerate(chunks):
            if c["char_start"] <= pdf_off < c["char_end"]:
                return i, c
        return -1, None

    print(f"  PDF has {len(pdf_chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  {'':1} {'json_off':>9}  {'pdf_off':>9}  {'chunk_idx':>9}  {'answer':<35}")
    print("  " + "─" * 80)
    for q in answered[:10]:
        a = q["answers"][0]
        j_off = a["answer_start"]
        j_text = a["text"]
        pdf_off, _ = map_json_to_pdf(
            j_off, len(json_ctx), j_anchors, p_anchors, pdf_text, j_text
        )
        cidx, chunk = find_chunk(pdf_off, pdf_chunks)
        print(f"  · {j_off:>9}  {pdf_off:>9}  {cidx:>9}  {repr(j_text[:33]):<35}")


if __name__ == "__main__":
    main()
