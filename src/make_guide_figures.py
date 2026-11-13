"""Make the two extra figures for READER_GUIDE.md.

Writes to results/figures/ (PNG only, the guide is Markdown):
  guide_domains  one random greenhouse and one random field image with their labelled boxes
  guide_iou      three pairs of boxes with their intersection over union (IoU)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

from analyze import load_ground_truth
from make_figures import CLASS_COLORS, COL_W, INK, MUTED

REPO = Path(__file__).resolve().parents[1]


def save_png(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


def fig_domains(data_dir, out_dir, seed=0, min_boxes=5):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.6))
    for ax, (split, title) in zip(axes, [("greenhouse_train", "greenhouse (training)"), ("field_test", "field (test)")]):
        ids, gt = load_ground_truth(data_dir, split)
        i = next(i for i in rng.permutation(ids) if len(gt[i][0]) >= min_boxes)
        with Image.open(data_dir / "images" / f"{i}.jpg") as im:
            im = im.convert("RGB")
        width = max(2, round(max(im.size) / 300))
        d = ImageDraw.Draw(im)
        for c, b in zip(*gt[i]):
            d.rectangle(list(b), outline=CLASS_COLORS[c], width=width)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}: {i}", fontsize=8, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=c, lw=2, label=n) for n, c in zip(["weed label", "crop label"], CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    save_png(fig, out_dir, "guide_domains")


def fig_iou(out_dir):
    # Label box fixed at (0, 0)-(4, 4); the prediction is shifted to give three typical overlaps.
    shifts = [0.2, 0.7, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.2))
    for ax, s in zip(axes, shifts):
        inter = (4 - s) ** 2
        iou = inter / (2 * 16 - inter)
        ax.add_patch(Rectangle((0, 0), 4, 4, fill=False, ec=CLASS_COLORS[0], lw=2))
        ax.add_patch(Rectangle((s, s), 4, 4, fill=False, ec="#ffd400", lw=2))
        ax.add_patch(Rectangle((s, s), 4 - s, 4 - s, color=MUTED, alpha=0.25, lw=0))
        ax.set_xlim(-0.5, 7.1)
        ax.set_ylim(7.1, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        verdict = "found" if iou >= 0.6 else "just found" if iou >= 0.5 else "not found"
        ax.set_title(f"IoU = {iou:.2f}: {verdict}", fontsize=8, color=INK)
    handles = [plt.Line2D([], [], color=CLASS_COLORS[0], lw=2, label="label (the right answer)"),
               plt.Line2D([], [], color="#ffd400", lw=2, label="prediction"),
               Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, label="overlap")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    save_png(fig, out_dir, "guide_iou")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = ap.parse_args()
    out_dir = args.results_dir / "figures"
    fig_domains(args.data_dir, out_dir)
    fig_iou(out_dir)


if __name__ == "__main__":
    main()
