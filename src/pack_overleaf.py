"""Pack the report into one zip for Overleaf (New Project > Upload Project).

The zip holds report/ (main.tex, style files, references.bib, tables/) plus the PDF figures from
results/figures/ under figures/, which main.tex finds through \\graphicspath.
Run src/make_figures.py, src/colour_stats.py and src/make_tables.py first.
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", type=Path, default=REPO / "report")
    ap.add_argument("--figures-dir", type=Path, default=REPO / "results/figures")
    ap.add_argument("--out", type=Path, default=REPO / "report_overleaf.zip")
    args = ap.parse_args()

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.report_dir.rglob("*")):
            if f.is_file() and f.suffix in {".tex", ".sty", ".bst", ".bib"}:
                z.write(f, f.relative_to(args.report_dir).as_posix())
        for f in sorted(args.figures_dir.glob("*.pdf")):
            z.write(f, f"figures/{f.name}")
        names = z.namelist()
    print(f"{args.out}: {len(names)} files")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
