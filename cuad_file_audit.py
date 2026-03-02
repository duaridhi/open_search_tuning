# ---------------------------------------------------------------------------
# cuad_file_audit.py
# ---------------------------------------------------------------------------
# Compares filenames across three sources:
#   1. master_clauses.csv  — the ground-truth list of contracts
#   2. full_contract_pdf/  — downloaded PDF files
#   3. full_contract_txt/  — downloaded TXT files
#
# Reports any discrepancies: missing files, extra files, stem mismatches.
#
# Usage (run directly):
#   python cuad_file_audit.py
#
# Or import and call:
#   from cuad_file_audit import run_audit
#   run_audit()
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root — adjust if needed)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
CUAD_V1_DIR   = _PROJECT_ROOT / "cuad_data" / "CUAD_v1"
CSV_PATH      = CUAD_V1_DIR / "master_clauses.csv"
PDF_DIR       = CUAD_V1_DIR / "full_contract_pdf"
TXT_DIR       = CUAD_V1_DIR / "full_contract_txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_filenames(csv_path: Path) -> set[str]:
    """Return the set of bare filenames from the 'Filename' column in the CSV."""
    filenames: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Filename", "").strip()
            if name:
                filenames.add(name)
    return filenames


def _collect_files(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    """
    Recursively collect files with the given suffixes under *directory*.

    Returns a dict mapping bare filename → full Path.
    Warns if duplicate bare filenames are found across subdirectories.
    """
    files: dict[str, Path] = {}
    duplicates: list[str] = []

    for suffix in suffixes:
        for p in directory.rglob(f"*{suffix}"):
            name = p.name
            if name in files:
                duplicates.append(f"  DUPLICATE: {name}\n    {files[name]}\n    {p}")
            else:
                files[name] = p

    if duplicates:
        print(f"\n⚠  Duplicate filenames found under {directory}:")
        for d in duplicates:
            print(d)

    return files


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def run_audit(
    csv_path: Path = CSV_PATH,
    pdf_dir: Path = PDF_DIR,
    txt_dir: Path = TXT_DIR,
) -> dict:
    """
    Compare filenames across the CSV, PDF folder, and TXT folder.

    Returns a dict with keys:
        csv_only_pdf_names   – in CSV but no matching PDF
        csv_only_txt_names   – in CSV but no matching TXT
        pdf_only             – PDFs not in CSV
        txt_only             – TXTs not in CSV (stem-matched to CSV pdf names)
        stem_mismatches      – PDFs in CSV whose .txt counterpart is missing/different
        summary              – human-readable summary string
    """
    print("=" * 70)
    print("CUAD FILE AUDIT")
    print("=" * 70)

    # --- Load all three sources -------------------------------------------
    csv_names = _read_csv_filenames(csv_path)          # e.g. "Contract.pdf"
    pdf_files = _collect_files(pdf_dir, (".pdf", ".PDF"))
    txt_files = _collect_files(txt_dir, (".txt", ".TXT"))

    # Stems from CSV (strip extension for cross-format comparison)
    csv_stems = {Path(n).stem: n for n in csv_names}   # stem → original csv name
    pdf_stems = {Path(n).stem: n for n in pdf_files}   # stem → bare pdf filename
    txt_stems = {Path(n).stem: n for n in txt_files}   # stem → bare txt filename

    print(f"\nSources:")
    print(f"  CSV entries   : {len(csv_names):>5}")
    print(f"  PDF files     : {len(pdf_files):>5}  (in {pdf_dir.relative_to(_PROJECT_ROOT)})")
    print(f"  TXT files     : {len(txt_files):>5}  (in {txt_dir.relative_to(_PROJECT_ROOT)})")

    # ---- 1. CSV vs PDFs --------------------------------------------------
    csv_only_pdf = {s: csv_stems[s] for s in csv_stems if s not in pdf_stems}
    pdf_only     = {s: pdf_stems[s] for s in pdf_stems if s not in csv_stems}

    # ---- 2. CSV vs TXTs --------------------------------------------------
    csv_only_txt = {s: csv_stems[s] for s in csv_stems if s not in txt_stems}
    txt_only     = {s: txt_stems[s] for s in txt_stems if s not in csv_stems}

    # ---- 3. PDF ↔ TXT stem parity ----------------------------------------
    # Contracts that have a PDF but are missing a TXT (and are in the CSV)
    pdf_no_txt = {s for s in pdf_stems if s in csv_stems and s not in txt_stems}
    txt_no_pdf = {s for s in txt_stems if s in csv_stems and s not in pdf_stems}

    # ---- Print results ----------------------------------------------------
    _section("1. In CSV but missing PDF", csv_only_pdf)
    _section("2. PDF files not in CSV (extra)", pdf_only)
    _section("3. In CSV but missing TXT", csv_only_txt)
    _section("4. TXT files not in CSV (extra)", txt_only)
    _section("5. Has PDF but no TXT counterpart", {s: pdf_stems[s] for s in pdf_no_txt})
    _section("6. Has TXT but no PDF counterpart", {s: txt_stems[s] for s in txt_no_pdf})

    # ---- Summary ----------------------------------------------------------
    summary_lines = [
        "\n" + "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"  CSV entries              : {len(csv_names)}",
        f"  PDF files                : {len(pdf_files)}",
        f"  TXT files                : {len(txt_files)}",
        "",
        f"  CSV entries missing PDF  : {len(csv_only_pdf)}",
        f"  Extra PDFs (not in CSV)  : {len(pdf_only)}",
        f"  CSV entries missing TXT  : {len(csv_only_txt)}",
        f"  Extra TXTs (not in CSV)  : {len(txt_only)}",
        f"  PDF without TXT pair     : {len(pdf_no_txt)}",
        f"  TXT without PDF pair     : {len(txt_no_pdf)}",
    ]

    all_ok = not any([csv_only_pdf, pdf_only, csv_only_txt, txt_only, pdf_no_txt, txt_no_pdf])
    summary_lines.append("")
    summary_lines.append("  ✓ All files match!" if all_ok else "  ⚠  Discrepancies found — see sections above.")
    summary_lines.append("=" * 70)

    summary = "\n".join(summary_lines)
    print(summary)

    return {
        "csv_only_pdf_names": csv_only_pdf,
        "csv_only_txt_names": csv_only_txt,
        "pdf_only": pdf_only,
        "txt_only": txt_only,
        "pdf_no_txt": pdf_no_txt,
        "txt_no_pdf": txt_no_pdf,
        "summary": summary,
    }


def _section(title: str, items: dict) -> None:
    print(f"\n--- {title} ({len(items)}) ---")
    if not items:
        print("  (none)")
    else:
        for stem, name in sorted(items.items())[:20]:
            print(f"  {name}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_audit()
