"""Make fixed train/val/test splits and write them to data/processed/splits/.

  greenhouse  -> train / val / test (~70/15/15 of images), split by groups of consecutive
                 image IDs: consecutive photos show the same planter, so splitting single
                 images would leak near-identical scenes between splits.
  field       -> test only (all images).
  uncertain   -> excluded.

Among many random group assignments (fixed seed) the one whose val and test splits are
closest to 15% of the greenhouse weed and crop boxes is kept, so both classes appear in
every split. Writes one image_id per line to <split>.txt and a summary to
results/splits_summary.csv.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def box_counts(ann_dir, image_id):
    names = [o.findtext("name") for o in ET.parse(ann_dir / f"{image_id}.xml").getroot().iter("object")]
    return names.count("weed"), names.count("crop")


def assign_groups(groups, n_trials, rng):
    """Return {group: split} whose image/box fractions best match FRACTIONS."""
    totals = groups[["n_images", "n_weed", "n_crop"]].sum()
    best, best_score = None, np.inf
    for _ in range(n_trials):
        order = rng.permutation(len(groups))
        cum = groups["n_images"].values[order].cumsum() / totals["n_images"]
        split = np.where(cum <= FRACTIONS["train"] + 1e-9, "train",
                         np.where(cum <= FRACTIONS["train"] + FRACTIONS["val"] + 1e-9, "val", "test"))
        assigned = pd.Series(split, index=groups.index[order])
        score = 0.0
        for name in ("val", "test"):
            part = groups.loc[assigned[assigned == name].index]
            for col in ("n_images", "n_weed", "n_crop"):
                score += (part[col].sum() / totals[col] - FRACTIONS[name]) ** 2
        if score < best_score:
            best, best_score = assigned, score
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/Ronin_OPEN_DB")
    ap.add_argument("--domains", type=Path, default=REPO / "data/processed/domains.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/processed/splits")
    ap.add_argument("--summary", type=Path, default=REPO / "results/splits_summary.csv")
    ap.add_argument("--block-size", type=int, default=12, help="consecutive greenhouse images per group")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.domains, dtype=str)
    df[["n_weed", "n_crop"]] = [box_counts(args.raw_dir / "annotations", i) for i in df["image_id"]]
    df["num"] = df["image_id"].str.extract(r"(\d+)$")[0].astype(int)

    gh = df[df["domain"] == "greenhouse"].sort_values("num").reset_index(drop=True)
    gh["group"] = gh.index // args.block_size
    groups = gh.groupby("group").agg(n_images=("image_id", "size"), n_weed=("n_weed", "sum"), n_crop=("n_crop", "sum"))
    gh["split"] = gh["group"].map(assign_groups(groups, args.trials, np.random.default_rng(args.seed)))

    splits = {f"greenhouse_{s}": gh.loc[gh["split"] == s] for s in FRACTIONS}
    splits["field_test"] = df[df["domain"] == "field"].sort_values("num")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, part in splits.items():
        (args.out_dir / f"{name}.txt").write_text("\n".join(part["image_id"]) + "\n")
        rows.append({"split": name, "n_images": len(part), "n_groups": part["group"].nunique() if "group" in part else "",
                     "n_weed": int(part["n_weed"].sum()), "n_crop": int(part["n_crop"].sum()),
                     "n_images_with_crop": int((part["n_crop"] > 0).sum())})
    summary = pd.DataFrame(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
