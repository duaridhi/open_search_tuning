"""
eval_dataset.py — pre-ingest PDF dataset evaluation

Samples a PDF directory, reports file-size and character-count statistics,
estimates chunk count under the current chunking parameters, and flags
OCR candidates (scanned/image-only PDFs with < 50 chars/page).

Usage
─────
    python tests/eval/eval_dataset.py [PDF_ROOT] [options]

    PDF_ROOT defaults to the path hardcoded in upload_to_qdrant.py.

Options
    --chunk-size INT      Characters per chunk (default: 500)
    --chunk-overlap INT   Overlap between chunks (default: 50)
    --step INT            Sample every Nth PDF (default: 10; use 1 for all)
    --ocr-threshold INT   Chars/page below which a PDF is an OCR candidate (default: 50)
"""

import argparse
import statistics
import sys
from pathlib import Path


DEFAULT_PDF_ROOT = Path(
    "/home/ridhi/projects/project1/open_search_tuning"
    "/cuad_opensearch/cuad_data/CUAD_v1/full_contract_pdf"
)
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_STEP          = 10
DEFAULT_OCR_THRESHOLD = 50


def build_extractor():
    try:
        import fitz

        def extract(path: Path):
            with fitz.open(str(path)) as doc:
                n_pages = len(doc)
                text = "\n\n".join(p.get_text("text") or "" for p in doc)
            return text, n_pages

        return "pymupdf", extract
    except ImportError:
        pass

    try:
        import pdfplumber

        def extract(path: Path):
            with pdfplumber.open(str(path)) as pdf:
                n_pages = len(pdf.pages)
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            return text, n_pages

        return "pdfplumber", extract
    except ImportError:
        pass

    raise RuntimeError("No PDF library found. Install pymupdf or pdfplumber.")


def file_size_stats(pdfs: list[Path]) -> dict:
    sizes = sorted(p.stat().st_size for p in pdfs)
    n = len(sizes)
    return {
        "count":      n,
        "min_kb":     sizes[0] / 1024,
        "median_kb":  sizes[n // 2] / 1024,
        "mean_kb":    sum(sizes) / n / 1024,
        "max_kb":     sizes[-1] / 1024,
        "total_mb":   sum(sizes) / 1024 / 1024,
    }


def run(
    pdf_root: Path,
    chunk_size: int,
    chunk_overlap: int,
    step: int,
    ocr_threshold: int,
):
    pdfs = sorted(p for p in pdf_root.rglob("*") if p.suffix.upper() == ".PDF")
    if not pdfs:
        print(f"ERROR: No PDFs found under {pdf_root}", file=sys.stderr)
        sys.exit(1)

    extractor_name, extract = build_extractor()
    sample = pdfs[::step]
    stride = chunk_size - chunk_overlap

    char_counts: list[int] = []
    empty_pdfs:  list[str] = []
    error_pdfs:  list[tuple[str, str]] = []
    ocr_candidates: list[str] = []

    for p in sample:
        try:
            text, n_pages = extract(p)
            chars_per_page = len(text.strip()) / max(n_pages, 1)
            if not text.strip():
                empty_pdfs.append(p.name)
                ocr_candidates.append(p.name)
            elif chars_per_page < ocr_threshold:
                ocr_candidates.append(p.name)
                char_counts.append(len(text))
            else:
                char_counts.append(len(text))
        except Exception as exc:
            error_pdfs.append((p.name, str(exc)))

    sizes = file_size_stats(pdfs)

    def est_chunks(n: int) -> int:
        return max(1, round(n / stride))

    chunk_counts = [est_chunks(c) for c in char_counts]
    mean_chunks  = statistics.mean(chunk_counts) if chunk_counts else 0
    med_chunks   = statistics.median(chunk_counts) if chunk_counts else 0

    sep = "─" * 45
    print(f"\nDataset: {pdf_root}")
    print(sep)
    print(f"{'PDFs':<26}: {sizes['count']}")
    print(f"{'Extractor':<26}: {extractor_name}")
    print(f"{'Empty / errored':<26}: {len(empty_pdfs)} / {len(error_pdfs)}")
    print(f"{'OCR candidates':<26}: {len(ocr_candidates)}  (< {ocr_threshold} chars/page)")
    print()
    print(f"PDF file sizes (all {sizes['count']} PDFs)")
    print(f"  {'min':<22}: {sizes['min_kb']:.0f} KB")
    print(f"  {'median':<22}: {sizes['median_kb']:.0f} KB")
    print(f"  {'mean':<22}: {sizes['mean_kb']:.0f} KB")
    print(f"  {'max':<22}: {sizes['max_kb']:.0f} KB")
    print(f"  {'total':<22}: {sizes['total_mb']:.1f} MB")

    if char_counts:
        print()
        print(f"Char count (sample of {len(sample)} PDFs — every {step}th)")
        print(f"  {'min':<22}: {min(char_counts):,}")
        print(f"  {'median':<22}: {statistics.median(char_counts):,.0f}")
        print(f"  {'mean':<22}: {statistics.mean(char_counts):,.0f}")
        print(f"  {'max':<22}: {max(char_counts):,}")
        print()
        print(f"Chunking params   : CHUNK_SIZE={chunk_size}  CHUNK_OVERLAP={chunk_overlap}  (stride={stride})")
        print(f"Chunks/doc        : median {med_chunks:.0f}  mean {mean_chunks:.0f}")

    print(sep)

    if char_counts:
        lo = round(len(pdfs) * min(med_chunks, mean_chunks))
        hi = round(len(pdfs) * max(med_chunks, mean_chunks))
        print(f"Estimated total   : ~{lo:,}–{hi:,} chunks")

    print(f"Ingest command    : QDRANT_COLLECTION=<name> python cuad-demo-quadrant/upload_to_qdrant.py")

    if empty_pdfs:
        print(f"\n⚠ Empty PDFs (no text layer — must OCR before ingest):")
        for name in empty_pdfs[:10]:
            print(f"    {name}")
        if len(empty_pdfs) > 10:
            print(f"    ... and {len(empty_pdfs) - 10} more")

    if ocr_candidates:
        print(f"\n⚠ Sparse-text PDFs (< {ocr_threshold} chars/page — consider ocrmypdf --force-ocr):")
        for name in [n for n in ocr_candidates if n not in empty_pdfs][:10]:
            print(f"    {name}")

    if error_pdfs:
        print(f"\n✗ Extraction errors:")
        for name, err in error_pdfs[:5]:
            print(f"    {name}: {err}")

    print()
    return len(ocr_candidates) + len(error_pdfs)  # exit-code hint: 0 = clean


def main():
    parser = argparse.ArgumentParser(
        description="Pre-ingest evaluation of a PDF dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_root",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF_ROOT,
        help="Root directory containing PDF files",
    )
    parser.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--step",          type=int, default=DEFAULT_STEP,
                        help="Sample every Nth PDF (1 = all)")
    parser.add_argument("--ocr-threshold", type=int, default=DEFAULT_OCR_THRESHOLD,
                        help="Chars/page below which a PDF is flagged as OCR candidate")
    args = parser.parse_args()

    issues = run(
        pdf_root=args.pdf_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        step=args.step,
        ocr_threshold=args.ocr_threshold,
    )
    sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
