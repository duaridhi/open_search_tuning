# ---------------------------------------------------------------------------
# cuad_fix_filenames.py
# ---------------------------------------------------------------------------
# Renames PDF and TXT files on disk to match the filenames in master_clauses.csv.
#
# Issues fixed:
#   1. Sanitised characters restored:  _ → '  (apostrophe)
#                                      _ → &  (ampersand)
#   2. Harpoon PDF: extra '_Option Agreement' suffix removed
#   3. NETGEAR TXT: trailing '-' added to match CSV / JSON title
#
# Usage:
#   python cuad_fix_filenames.py           # dry run (shows plan, no changes)
#   python cuad_fix_filenames.py --apply   # actually rename files
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

DRY_RUN = "--apply" not in sys.argv

BASE = Path(__file__).resolve().parent / "cuad_data" / "CUAD_v1"

# ---------------------------------------------------------------------------
# Rename map  →  (current_relative_path, new_bare_filename)
# ---------------------------------------------------------------------------
RENAMES: list[tuple[str, str]] = [

    # ── PDFs ──────────────────────────────────────────────────────────────
    (
        "full_contract_pdf/Part_III/Joint Venture _ Filing/"
        "MACY_S,INC_05_11_2020-EX-99.4-JOINT FILING AGREEMENT.PDF",
        "MACY'S,INC_05_11_2020-EX-99.4-JOINT FILING AGREEMENT.PDF",
    ),
    (
        "full_contract_pdf/Part_I/Strategic Alliance/"
        "MOELIS_CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT.PDF",
        "MOELIS&CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT.PDF",
    ),
    (
        "full_contract_pdf/Part_III/Marketing/"
        "Monsanto Company - SECOND A_R EXCLUSIVE AGENCY AND MARKETING AGREEMENT .PDF",
        "Monsanto Company - SECOND A&R EXCLUSIVE AGENCY AND MARKETING AGREEMENT .PDF",
    ),
    (
        "full_contract_pdf/Part_III/Marketing/"
        "PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .PDF",
        "PACIRA PHARMACEUTICALS, INC. - A&R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .PDF",
    ),
    (
        "full_contract_pdf/Part_I/Strategic Alliance/"
        "PLAYAHOTELS_RESORTSNV_03_14_2017-EX-10.22-STRATEGIC ALLIANCE AGREEMENT (Hyatt Ziva Cancun).PDF",
        "PLAYAHOTELS&RESORTSNV_03_14_2017-EX-10.22-STRATEGIC ALLIANCE AGREEMENT (Hyatt Ziva Cancun).PDF",
    ),
    (
        "full_contract_pdf/Part_III/Marketing/"
        "Reinsurance Group of America, Incorporated - A_R REMARKETING  AGREEMENT.PDF",
        "Reinsurance Group of America, Incorporated - A&R REMARKETING  AGREEMENT.PDF",
    ),
    (
        "full_contract_pdf/Part_III/Marketing/"
        "SightLife Surgical, Inc. - STRATEGIC SALES _ MARKETING AGREEMENT.PDF",
        "SightLife Surgical, Inc. - STRATEGIC SALES & MARKETING AGREEMENT.PDF",
    ),
    # Harpoon: remove the extra '_Option Agreement' suffix
    (
        "full_contract_pdf/Part_I/Development/"
        "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement_Option Agreement.pdf",
        "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement.pdf",
    ),

    # ── TXTs ──────────────────────────────────────────────────────────────
    (
        "full_contract_txt/Part_II/"
        "PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .txt",
        "PACIRA PHARMACEUTICALS, INC. - A&R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .txt",
    ),
    (
        "full_contract_txt/Part_II/"
        "PLAYAHOTELS_RESORTSNV_03_14_2017-EX-10.22-STRATEGIC ALLIANCE AGREEMENT (Hyatt Ziva Cancun).txt",
        "PLAYAHOTELS&RESORTSNV_03_14_2017-EX-10.22-STRATEGIC ALLIANCE AGREEMENT (Hyatt Ziva Cancun).txt",
    ),
    (
        "full_contract_txt/Part_II/"
        "SightLife Surgical, Inc. - STRATEGIC SALES _ MARKETING AGREEMENT.txt",
        "SightLife Surgical, Inc. - STRATEGIC SALES & MARKETING AGREEMENT.txt",
    ),
    # NETGEAR TXT: add missing trailing '-' to match CSV/JSON title
    (
        "full_contract_txt/Part_II/"
        "NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR.txt",
        "NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR-.txt",
    ),
]


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def main() -> None:
    mode = "DRY RUN" if DRY_RUN else "APPLYING RENAMES"
    print("=" * 70)
    print(f"CUAD FILENAME FIX  [{mode}]")
    print("=" * 70)
    if DRY_RUN:
        print("Pass --apply to actually rename files.\n")

    ok = skipped = errors = 0

    for rel_old, new_name in RENAMES:
        src = BASE / rel_old
        dst = src.parent / new_name

        if not src.exists():
            print(f"  ✗ NOT FOUND: {src.relative_to(BASE)}")
            errors += 1
            continue

        if dst.exists():
            print(f"  ⚠ TARGET EXISTS (skip): {dst.relative_to(BASE)}")
            skipped += 1
            continue

        print(f"  {'[dry]' if DRY_RUN else '✓'} {src.name}")
        print(f"       → {new_name}")

        if not DRY_RUN:
            src.rename(dst)

        ok += 1

    print()
    print(f"Result: {ok} rename(s), {skipped} skipped, {errors} not found")
    print("=" * 70)


if __name__ == "__main__":
    main()
