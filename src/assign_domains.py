"""Assign each raw image to a domain and write data/processed/domains.csv.

Rule (decided by visual inspection of contact sheets, see results/inspection/NOTES.md):
  IMG_*.JPG             -> greenhouse  (pots/planters under LED light; Canon EOS 800D per Sudars et al. 2020)
  numeric id >= 10000   -> field       (bare soil, 1280x720; RealSense D435 per Sudars et al. 2020)
  numeric id <  1000    -> uncertain   (1280x720 like the field set, but peat substrate with black pot
                                        edges; not clearly either domain, excluded by default)
"""

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def domain_of(filename):
    if re.fullmatch(r"IMG_\d+\.jpe?g", filename, re.IGNORECASE):
        return "greenhouse"
    m = re.fullmatch(r"(\d+)\.jpe?g", filename, re.IGNORECASE)
    if m:
        return "field" if int(m.group(1)) >= 10000 else "uncertain"
    raise ValueError(f"Unexpected filename: {filename}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/Ronin_OPEN_DB")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/domains.csv")
    args = ap.parse_args()

    files = sorted(p.name for p in (args.raw_dir / "raw images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    df = pd.DataFrame({"image_id": [Path(f).stem for f in files], "file": files})
    df["domain"] = df["file"].map(domain_of)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df["domain"].value_counts().to_string())


if __name__ == "__main__":
    main()
