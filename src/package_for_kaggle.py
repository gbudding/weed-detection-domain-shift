"""Zip the YOLO dataset (and optionally the code) for upload as Kaggle datasets.

Writes to data/kaggle/:
  weed-ds-yolo.zip   contents of data/processed/yolo/ (images, labels, split lists, YAMLs)
  weed-ds-code.zip   src/ + requirements.txt, only needed if the notebook can't git-clone the repo
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def zip_dir(zf, root, arc_prefix="", skip_suffixes=(".cache",)):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix not in skip_suffixes and "__pycache__" not in p.parts:
            zf.write(p, arc_prefix + p.relative_to(root).as_posix())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/kaggle")
    ap.add_argument("--code", action="store_true", help="also write weed-ds-code.zip")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Images are already JPEG-compressed, so store them without recompression.
    with zipfile.ZipFile(args.out_dir / "weed-ds-yolo.zip", "w", zipfile.ZIP_STORED) as zf:
        zip_dir(zf, args.yolo_dir)
    if args.code:
        with zipfile.ZipFile(args.out_dir / "weed-ds-code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zip_dir(zf, REPO / "src", "src/")
            zf.write(REPO / "requirements.txt", "requirements.txt")
    for f in sorted(args.out_dir.glob("*.zip")):
        print(f"{f}  {f.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
