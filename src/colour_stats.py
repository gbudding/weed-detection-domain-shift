"""Compare the colour of greenhouse (training) and field (test) images, and relate it to the HSV augmentation.

Images are resized to a long side of --imgsz (as in training) and converted to HSV with OpenCV, the same
colour space Ultralytics augments in (H 0-179, S and V 0-255). Pixels are taken from the whole image and
from inside the weed boxes, split into plant pixels (excess green > 0.1) and soil pixels: weed boxes are
small and mostly soil. Hue only counts pixels with saturation >= 40 (hue is undefined for grey pixels).

Ultralytics' RandomHSV draws per image: hue shift +-hsv_h*180, saturation and value scaled by 1 +- hsv_s,
1 +- hsv_v. The figure shades the range the default and strong settings reach around the greenhouse median.

Writes:
  results/colour_per_image.csv   per image and region: median H, S, V
  results/colour_summary.csv     per domain and region: median of the per-image medians, and field vs greenhouse
  results/figures/colour_hist.{pdf,png}
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load_ground_truth
from make_figures import COL_W, GRID, INK, MUTED, save, style_axis
from train import HSV

REPO = Path(__file__).resolve().parents[1]
DOMAINS = {"greenhouse_train": "greenhouse", "field_test": "field"}
DOMAIN_COLORS = {"greenhouse": "#1baf7a", "field": "#6250d6"}  # categorical slots 3 and 7: not the model colours
CHANNELS = [("H", 180), ("S", 256), ("V", 256)]
MIN_SAT_FOR_HUE = 40
EXG_PLANT = 0.1  # excess green (2g - r - b on chromatic coordinates) above which a pixel counts as plant
REGIONS = ("image", "plant", "soil")


def image_stats(data_dir, split, imgsz):
    """Per-image median H, S, V and pixel histograms for the whole image and the plant and soil pixels in weed boxes."""
    ids, gt = load_ground_truth(data_dir, split)
    hist = {r: [np.zeros(n) for _, n in CHANNELS] for r in REGIONS}
    rows = []
    for i in ids:
        img = cv2.imread(str(data_dir / "images" / f"{i}.jpg"))
        f = imgsz / max(img.shape[:2])
        hsv = cv2.cvtColor(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], bool)
        for c, b in zip(*gt[i]):
            if c == 0:
                x0, y0, x1, y1 = np.round(b * f).astype(int)
                mask[max(y0, 0):y1, max(x0, 0):x1] = True
        # Weed boxes are mostly soil: split their pixels into plant and soil by excess green.
        bgr = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(float)
        exg = (2 * bgr[..., 1] - bgr[..., 2] - bgr[..., 0]) / (bgr.sum(2) + 1e-6)
        plant = exg > EXG_PLANT
        regions = (("image", hsv.reshape(-1, 3)), ("plant", hsv[mask & plant]), ("soil", hsv[mask & ~plant]))
        for region, px in regions:
            if len(px) == 0:
                continue
            # Hue is undefined for grey pixels, so it is only counted where saturation >= MIN_SAT_FOR_HUE.
            hue = px[px[:, 1] >= MIN_SAT_FOR_HUE, 0]
            for k, (_, n) in enumerate(CHANNELS):
                hist[region][k] += np.bincount(hue if k == 0 else px[:, k], minlength=n)
            rows.append({"image_id": i, "domain": DOMAINS[split], "region": region, "n_pixels": len(px),
                         "H": float(np.median(hue)) if len(hue) else np.nan,
                         "S": float(np.median(px[:, 1])), "V": float(np.median(px[:, 2]))})
    return pd.DataFrame(rows), hist


def summarise(per_image):
    med = per_image.groupby(["region", "domain"])[["H", "S", "V"]].median()
    rows = []
    for region in REGIONS:
        g, f = med.loc[(region, "greenhouse")], med.loc[(region, "field")]
        rows.append({"region": region, **{f"greenhouse_{c}": g[c] for c in "HSV"}, **{f"field_{c}": f[c] for c in "HSV"},
                     "H_shift": f["H"] - g["H"], "S_ratio": f["S"] / g["S"], "V_ratio": f["V"] / g["V"]})
    return pd.DataFrame(rows)


def fig_colour(hists, summary, out_dir):
    g = summary.set_index("region").loc["plant"]
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 1.6))
    for k, (ax, (ch, n)) in enumerate(zip(axes, CHANNELS)):
        centre = g[f"greenhouse_{ch}"]
        # Range reached by the augmentation around the greenhouse plant median (strong drawn first, underneath).
        for setting, shade in (("strong", "#efeeea"), ("default", "#dcdad3")):
            gain = HSV[setting][f"hsv_{ch.lower()}"]
            lo, hi = (centre - gain * 180, centre + gain * 180) if ch == "H" else (centre * (1 - gain), centre * (1 + gain))
            ax.axvspan(max(lo, 0), min(hi, n - 1), color=shade, lw=0, zorder=0)
        for domain, color in DOMAIN_COLORS.items():
            for region, ls in (("plant", "-"), ("soil", ":")):
                h = hists[domain][region][k]
                ax.plot(np.arange(n), h / h.sum(), color=color, ls=ls, lw=1.2 if region == "plant" else 1)
        ax.set_xlim(0, 100 if ch == "H" else n - 1)  # hue: the soil-to-green range; above 100 is nearly empty
        ax.set_yticks([])
        ax.set_xlabel({"H": f"hue (0-179; S $\\geq$ {MIN_SAT_FOR_HUE})", "S": "saturation", "V": "value (brightness)"}[ch])
        style_axis(ax)
        ax.grid(False)
    axes[0].set_ylabel("share of pixels")
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=d) for d, c in DOMAIN_COLORS.items()]
    handles += [plt.Line2D([], [], color=MUTED, lw=1.2, label="plant pixels in weed boxes"),
                plt.Line2D([], [], color=MUTED, lw=1, ls=":", label="soil pixels in weed boxes"),
                plt.Rectangle((0, 0), 1, 1, color="#dcdad3", label="HSV default range"),
                plt.Rectangle((0, 0), 1, 1, color="#efeeea", label="HSV strong range")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 1.12),
               handlelength=1.5, columnspacing=1.0)
    save(fig, out_dir, "colour_hist")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    frames, hists = [], {}
    for split, domain in DOMAINS.items():
        df, hists[domain] = image_stats(args.data_dir, split, args.imgsz)
        frames.append(df)
        print(f"{split}: {df['image_id'].nunique()} images")
    per_image = pd.concat(frames)
    per_image.to_csv(args.results_dir / "colour_per_image.csv", index=False)
    summary = summarise(per_image)
    summary.to_csv(args.results_dir / "colour_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    fig_colour(hists, summary, args.results_dir / "figures")


if __name__ == "__main__":
    main()
