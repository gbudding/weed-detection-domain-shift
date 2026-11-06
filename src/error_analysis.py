"""Classify detection errors per run and test set, and save example crops per error type.

Detections with confidence >= --conf (fixed operating point, default 0.25) are matched to the
ground truth in descending confidence order. Each detection gets one type:
  tp          IoU >= 0.5 with an unmatched ground-truth box of the same class
  duplicate   IoU >= 0.5 with a same-class box that was already matched
  class       IoU >= 0.5 with a box of the other class (crop <-> weed confusion)
  localisation  0.1 <= IoU < 0.5 with a box of the same class
  background  IoU < 0.1 with every ground-truth box
  other       anything else (wrong class and poor overlap)
Each ground-truth box not matched by a tp is a false negative:
  missed      no detection of any class with IoU >= 0.1 (the object was not found at all)
  fn_covered  some detection overlaps it with IoU >= 0.1 (e.g. a localisation or class error)

Object size = side length (sqrt of area) in pixels after resizing the image to --imgsz:
tiny < 16, small 16-32, medium >= 32.

Writes to --out-dir:
  errors_per_run.csv       counts per run, target, type and size
  error_examples/<run>/<target>_<type>.jpg   grids of example crops (for --example-runs)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze import CLASSES, TARGETS, box_iou, load_ground_truth

REPO = Path(__file__).resolve().parents[1]
DET_TYPES = ["tp", "duplicate", "class", "localisation", "background", "other"]
GT_TYPES = ["missed", "fn_covered"]
SIZE_BINS = [(0, 16, "tiny"), (16, 32, "small"), (32, np.inf, "medium")]
CLASS_COLORS = ["#eb6834", "#2a78d6"]  # weed, crop (as in the inspection figures)
ERROR_COLOR = "#ffd400"  # yellow: distinct from the orange weed and blue crop labels


def size_bin(box, img_wh, imgsz):
    side = np.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0)) * imgsz / max(img_wh)
    return next(name for lo, hi, name in SIZE_BINS if lo <= side < hi)


def classify_image(gt_cls, gt_box, det_cls, det_box):
    """Return the type of each detection (in the given confidence order), the index of the ground-truth
    box each tp matched (else -1), and the type of each ground-truth box."""
    det_types, det_match = [], []
    matched = np.zeros(len(gt_cls), bool)
    iou = box_iou(gt_box, det_box) if len(gt_cls) and len(det_cls) else np.zeros((len(gt_cls), len(det_cls)))
    for j in range(len(det_cls)):
        col = iou[:, j]
        same = gt_cls == det_cls[j]
        free_same = np.where(same & ~matched, col, 0)
        if free_same.size and free_same.max() >= 0.5:
            k = free_same.argmax()
            matched[k] = True
            det_types.append("tp")
            det_match.append(k)
            continue
        det_match.append(-1)
        if np.any(same & matched & (col >= 0.5)):
            det_types.append("duplicate")
        elif np.any(~same & (col >= 0.5)):
            det_types.append("class")
        elif np.any(same & (col >= 0.1)):
            det_types.append("localisation")
        elif col.size == 0 or col.max() < 0.1:
            det_types.append("background")
        else:
            det_types.append("other")
    near = (iou >= 0.1).any(1) if iou.size else np.zeros(len(gt_cls), bool)
    gt_types = np.where(matched, "tp", np.where(near, "fn_covered", "missed"))
    return det_types, det_match, gt_types


def load_detections(pred_json, conf):
    preds = pd.DataFrame(json.loads(Path(pred_json).read_text()))
    preds = preds[preds["score"] >= conf].copy()
    preds["image_id"] = preds["image_id"].astype(str)
    preds["cls"] = preds["category_id"] - 1  # Ultralytics writes category_id = class + 1
    xywh = np.array(preds["bbox"].tolist()).reshape(-1, 4)
    preds[["x0", "y0", "x1", "y1"]] = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
    return preds.sort_values("score", ascending=False)


def analyse_run(run_dir, target, ids, gt, sizes, conf, imgsz):
    """Return one record per detection and per ground-truth box with its error type."""
    preds = load_detections(run_dir / f"eval_{target}/predictions.json", conf)
    by_image = dict(tuple(preds.groupby("image_id")))
    records = []
    for i in ids:
        gt_cls, gt_box = gt[i]
        p = by_image.get(i, preds.iloc[:0])
        det_box = p[["x0", "y0", "x1", "y1"]].values
        det_types, det_match, gt_types = classify_image(gt_cls, gt_box, p["cls"].values, det_box)
        for (_, d), t, k, b in zip(p.iterrows(), det_types, det_match, det_box):
            # A tp is binned by the size of the object it found, so tp / gt_total is recall per size.
            size = size_bin(gt_box[k] if k >= 0 else b, sizes[i], imgsz)
            records.append({"image_id": i, "kind": "det", "type": t, "cls": CLASSES[int(d["cls"])], "score": d["score"],
                            "size": size, "box": b.round(1).tolist()})
        for c, t, b in zip(gt_cls, gt_types, gt_box):
            if t != "tp":
                records.append({"image_id": i, "kind": "gt", "type": t, "cls": CLASSES[c], "score": np.nan,
                                "size": size_bin(b, sizes[i], imgsz), "box": b.round(1).tolist()})
    return pd.DataFrame(records)


def example_grid(records, data_dir, gt, error_type, out_path, n=24, cols=6, tile=200, seed=0):
    """Crops around example errors: ground truth in class colours, the error box in yellow."""
    rows = records[records["type"] == error_type]
    if rows.empty:
        return
    # False positives: the most confident ones (most harmful). Misses: a random sample.
    rows = rows.nlargest(n, "score") if error_type in DET_TYPES else rows.sample(min(n, len(rows)), random_state=seed)
    sheet = Image.new("RGB", (cols * tile, int(np.ceil(len(rows) / cols)) * tile), "white")
    for k, (_, r) in enumerate(rows.iterrows()):
        with Image.open(data_dir / "images" / f"{r['image_id']}.jpg") as im:
            im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = r["box"]
        half = max(48, 2 * max(x1 - x0, y1 - y0))  # context around the box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        crop = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        lw = max(1, int(half / 60))
        for c, b in zip(*gt[r["image_id"]]):
            d.rectangle(b.tolist(), outline=CLASS_COLORS[c], width=lw)
        d.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=ERROR_COLOR, width=lw * 2)
        im = im.crop(crop).resize((tile - 4, tile - 4))
        label = f"{r['cls']} {r['score']:.2f}" if r["kind"] == "det" else f"missed {r['cls']} ({r['size']})"
        ImageDraw.Draw(im).text((4, 2), label, fill="white", stroke_width=2, stroke_fill="black")
        sheet.paste(im, ((k % cols) * tile + 2, (k // cols) * tile + 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (operating point)")
    ap.add_argument("--example-runs", nargs="*", default=None,
                    help="runs to save example crops for (default: every seed-0 run)")
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.runs_dir.iterdir()
                      if (d / "run_config.json").exists() and all((d / f"eval_{t}/predictions.json").exists() for t in TARGETS))
    gts, sizes = {}, {}
    for t in TARGETS:
        ids, gt = load_ground_truth(args.data_dir, t)
        gts[t] = (ids, gt)
        for i in ids:
            with Image.open(args.data_dir / "images" / f"{i}.jpg") as im:
                sizes[i] = im.size

    rows = []
    for d in run_dirs:
        cfg = json.loads((d / "run_config.json").read_text())
        examples = d.name in args.example_runs if args.example_runs is not None else cfg.get("seed") == 0
        for t in TARGETS:
            rec = analyse_run(d, t, *gts[t], sizes, args.conf, cfg["imgsz"])
            n_gt = sum(len(g[0]) for g in gts[t][1].values())
            counts = rec.groupby(["cls", "size", "type"]).size()
            for (cls, size, typ), n in counts.items():
                rows.append({"run": d.name, "model": cfg["model"], "imgsz": cfg["imgsz"], "hsv": cfg["hsv"],
                             "epochs": cfg["epochs"], "seed": cfg["seed"], "target": t, "cls": cls, "size": size,
                             "type": typ, "count": int(n)})
            # Ground-truth totals per class and size, so recall per size can be computed from the CSV.
            gt_sizes = pd.Series([(CLASSES[c], size_bin(b, sizes[i], cfg["imgsz"]))
                                  for i in gts[t][0] for c, b in zip(*gts[t][1][i])]).value_counts()
            for (cls, size), n in gt_sizes.items():
                rows.append({"run": d.name, "model": cfg["model"], "imgsz": cfg["imgsz"], "hsv": cfg["hsv"],
                             "epochs": cfg["epochs"], "seed": cfg["seed"], "target": t, "cls": cls, "size": size,
                             "type": "gt_total", "count": int(n)})
            if examples:
                for typ in DET_TYPES[1:] + GT_TYPES[:1]:
                    example_grid(rec, args.data_dir, gts[t][1], typ, args.out_dir / "error_examples" / d.name / f"{t}_{typ}.jpg")
            summary = rec[rec["kind"] == "det"]["type"].value_counts().to_dict()
            missed = int((rec["type"] == "missed").sum())
            print(f"{d.name} {t}: {n_gt} gt, missed {missed}, detections {summary}")
    pd.DataFrame(rows).to_csv(args.out_dir / "errors_per_run.csv", index=False)


if __name__ == "__main__":
    main()
