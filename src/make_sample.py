"""Copy a small random sample of the raw dataset into sample_data/ (committed to the repo).

The sample keeps the raw layout (Pascal VOC annotations + images) and the fixed splits, restricted to the
sampled images, so the whole pipeline (conversion, training, evaluation) runs on it: see reproduce.sh.

Writes sample_data/Ronin_OPEN_DB/{annotations,raw images}/ and sample_data/splits/*.txt
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_PER_SPLIT = {"greenhouse_train": 16, "greenhouse_val": 4, "greenhouse_test": 4, "field_test": 16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/Ronin_OPEN_DB")
    ap.add_argument("--splits-dir", type=Path, default=REPO / "data/processed/splits")
    ap.add_argument("--out-dir", type=Path, default=REPO / "sample_data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_images = {p.stem: p for p in (args.raw_dir / "raw images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    for sub in ("annotations", "raw images"):
        (args.out_dir / "Ronin_OPEN_DB" / sub).mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits").mkdir(parents=True, exist_ok=True)
    total = 0
    for split, n in N_PER_SPLIT.items():
        ids = sorted(rng.choice((args.splits_dir / f"{split}.txt").read_text().split(), n, replace=False))
        (args.out_dir / "splits" / f"{split}.txt").write_text("".join(f"{i}\n" for i in ids))
        for i in ids:
            shutil.copyfile(args.raw_dir / "annotations" / f"{i}.xml", args.out_dir / "Ronin_OPEN_DB/annotations" / f"{i}.xml")
            shutil.copyfile(raw_images[i], args.out_dir / "Ronin_OPEN_DB/raw images" / raw_images[i].name)
            total += raw_images[i].stat().st_size
        print(f"{split}: {len(ids)} images")
    print(f"{total / 1e6:.1f} MB of images -> {args.out_dir}")


if __name__ == "__main__":
    main()
