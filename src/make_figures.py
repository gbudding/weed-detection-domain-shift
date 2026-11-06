"""Make the report figures from results/*.csv and the run folders.

Writes to results/figures/ (PDF for LaTeX + PNG for viewing):
  domain_gap      weed AP50 on greenhouse test (hollow) and field test (filled) per model and
                  setting, mean over seeds with 95% bootstrap CI; the connecting line is the gap
  training_curves greenhouse-val mAP50 per epoch for the baseline runs (one line per seed)
  recall_by_size  weed recall at conf 0.25 per object size, greenhouse vs field, per model
  qualitative     the same field crops: ground truth, YOLO and RT-DETR predictions (seed-0 baselines)

Run src/analyze.py and src/error_analysis.py first.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze import load_ground_truth

REPO = Path(__file__).resolve().parents[1]
# Categorical slots 1-2 of the reference palette, in fixed order.
MODEL_COLORS = {"yolo11s": "#2a78d6", "rtdetr-l": "#eb6834"}
MODEL_LABELS = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-l"}
CLASS_COLORS = ["#eb6834", "#2a78d6"]  # weed, crop (as in the inspection figures)
SETTINGS = [((640, "default"), "baseline"), ((1024, "default"), "1024 px"),
            ((640, "off"), "HSV off"), ((640, "strong"), "HSV strong"), ((640, "hue"), "full hue")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200})
COL_W = 3.25  # ICML single-column width (inches)


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_domain_gap(summary, epochs, out_dir, metric="weed_AP50"):
    s = summary[(summary["metric"] == metric) & (summary["epochs"] == epochs)]
    settings = [(key, label) for key, label in SETTINGS if not s[(s["imgsz"] == key[0]) & (s["hsv"] == key[1])].empty]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    offsets = {"yolo11s": -0.14, "rtdetr-l": 0.14}
    for x, ((imgsz, hsv), _) in enumerate(settings):
        for model, dx in offsets.items():
            rows = s[(s["model"] == model) & (s["imgsz"] == imgsz) & (s["hsv"] == hsv)].set_index("target")
            if rows.empty:
                continue
            c = MODEL_COLORS[model]
            gh, fi = rows.loc["greenhouse_test"], rows.loc["field_test"]
            ax.plot([x + dx] * 2, [gh["mean"], fi["mean"]], color=c, linewidth=1.5, alpha=0.5, zorder=1)
            for r, fill in ((gh, "white"), (fi, c)):
                ax.errorbar(x + dx, r["mean"], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                            fmt="o", ms=5, mfc=fill, mec=c, mew=1.5, ecolor=c, elinewidth=1, capsize=0, zorder=2)
    ax.set_xticks(range(len(settings)), [label.replace(" ", "\n") for _, label in settings], fontsize=7)
    ax.set_xlim(-0.5, len(settings) - 0.5)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    style_axis(ax)
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], marker="o", ms=5, lw=0, label=MODEL_LABELS[m]) for m in offsets]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", ms=5, mfc="white", lw=0, label="greenhouse test"),
                plt.Line2D([], [], color=MUTED, marker="o", ms=5, lw=0, label="field test")]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, out_dir, "domain_gap")


def fig_training_curves(runs_dir, runs, epochs, out_dir):
    base = runs[(runs["epochs"] == epochs) & (runs["imgsz"] == 640) & (runs["hsv"] == "default")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for _, r in base.iterrows():
        res = pd.read_csv(runs_dir / r["run"] / "results.csv")
        ax.plot(res["epoch"], res["metrics/mAP50(B)"], color=MODEL_COLORS[r["model"]], linewidth=1, alpha=0.8)
    for m in [m for m in MODEL_COLORS if m in set(base["model"])]:
        ax.plot([], [], color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    ax.set_xlabel("epoch")
    ax.set_ylabel("greenhouse val mAP50")
    ax.set_ylim(0, 1)
    style_axis(ax)
    ax.legend(loc="lower right")
    save(fig, out_dir, "training_curves")


def fig_recall_by_size(errors, epochs, out_dir):
    e = errors[(errors["epochs"] == epochs) & (errors["imgsz"] == 640) & (errors["hsv"] == "default") & (errors["cls"] == "weed")]
    counts = e.pivot_table(index=["model", "seed", "target", "size"], columns="type", values="count", aggfunc="sum", fill_value=0)
    recall = (counts["tp"] / counts["gt_total"]).groupby(["model", "target", "size"]).mean()
    sizes = ["tiny", "small", "medium"]
    size_labels = ["<16", "16–32", "≥32"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    models = [m for m in MODEL_COLORS if m in recall.index.get_level_values(0)]
    width = 0.8 / len(models)
    for ax, target in zip(axes, ["greenhouse_test", "field_test"]):
        for k, m in enumerate(models):
            vals = [recall.get((m, target, s), np.nan) for s in sizes]
            ax.bar(np.arange(3) + (k - (len(models) - 1) / 2) * width, vals, width * 0.92,
                   color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ax.set_xticks(range(3), size_labels, fontsize=7)
        ax.set_xlabel("object size (px)", fontsize=7)
        ax.set_title(target.replace("_test", ""), fontsize=8, loc="left", color=INK)
        style_axis(ax)
    axes[0].set_ylabel("weed recall (conf 0.25)")
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=7, bbox_to_anchor=(0.55, 1.08))
    save(fig, out_dir, "recall_by_size")


def draw_boxes(im, boxes_cls, width):
    d = ImageDraw.Draw(im)
    for b, c in boxes_cls:
        d.rectangle(list(b), outline=CLASS_COLORS[c], width=width)
    return im


def fig_qualitative(runs_dir, runs, data_dir, epochs, out_dir, n_images=2, conf=0.25, seed=0):
    base = runs[(runs["epochs"] == epochs) & (runs["imgsz"] == 640) & (runs["hsv"] == "default") & (runs["seed"] == 0)]
    run_by_model = {m: base[base["model"] == m]["run"].iloc[0] for m in MODEL_COLORS if (base["model"] == m).any()}
    ids, gt = load_ground_truth(data_dir, "field_test")
    rng = np.random.default_rng(seed)
    chosen = [i for i in rng.permutation(ids) if len(gt[i][0]) >= 5][:n_images]
    preds = {}
    for m, run in run_by_model.items():
        p = pd.DataFrame(json.loads((runs_dir / run / "eval_field_test/predictions.json").read_text()))
        p["image_id"] = p["image_id"].astype(str)
        preds[m] = p[p["score"] >= conf]
    columns = ["ground truth"] + [MODEL_LABELS[m] for m in run_by_model]
    panel = COL_W / len(columns)  # square panels filling one column
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(COL_W, panel * len(chosen) + 0.4), squeeze=False)
    for row, i in enumerate(chosen):
        with Image.open(data_dir / "images" / f"{i}.jpg") as im:
            im = im.convert("RGB")
        # Crop a square around the ground-truth weeds so the small boxes stay visible.
        centre = gt[i][1][:, :2].mean(0) + (gt[i][1][:, 2:] - gt[i][1][:, :2]).mean(0) / 2
        half = 150
        cx, cy = np.clip(centre, half, np.array(im.size) - half)
        crop = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        panels = [draw_boxes(im.copy(), zip(gt[i][1], gt[i][0]), 4)]
        for m in run_by_model:
            p = preds[m][preds[m]["image_id"] == i]
            boxes = [((x, y, x + w, y + h), int(c) - 1) for (x, y, w, h), c in zip(p["bbox"], p["category_id"])]
            panels.append(draw_boxes(im.copy(), boxes, 4))
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel.crop(crop))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=8, color=INK)
        axes[row, 0].set_ylabel(i, fontsize=7, color=MUTED)
    handles = [plt.Line2D([], [], color=c, lw=2, label=n) for n, c in zip(["weed", "crop"], CLASS_COLORS)]
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.07, wspace=0.03, hspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, "qualitative")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--epochs", type=int, default=100, help="which runs to plot (pilot runs have 5)")
    args = ap.parse_args()
    out_dir = args.results_dir / "figures"

    summary = pd.read_csv(args.results_dir / "config_summary.csv")
    runs = pd.read_csv(args.results_dir / "run_metrics.csv")
    errors = pd.read_csv(args.results_dir / "errors_per_run.csv")
    fig_domain_gap(summary, args.epochs, out_dir)
    fig_training_curves(args.runs_dir, runs, args.epochs, out_dir)
    fig_recall_by_size(errors, args.epochs, out_dir)
    fig_qualitative(args.runs_dir, runs, args.data_dir, args.epochs, out_dir)


if __name__ == "__main__":
    main()
