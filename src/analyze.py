"""Collect all runs and compute metrics with bootstrap confidence intervals.

For every run folder in --runs-dir (with run_config.json and eval_<target>/predictions.json)
AP is recomputed from the saved predictions, using the same matching and 101-point AP as
Ultralytics (so full-set numbers agree with metrics.json). Test images are then resampled
with replacement (--n-boot times, fixed seed); the same resamples are used for every run,
so differences between runs are paired.

Writes to --out-dir:
  run_metrics.csv     one row per run: AP per target and class (full test sets)
  config_summary.csv  one row per config (model, imgsz, hsv, epochs) and metric: mean and std
                      over seeds, 95% bootstrap CI over test images, for greenhouse test,
                      field test and the domain gap (greenhouse - field)
  comparisons.csv     paired differences: RT-DETR vs YOLO at equal settings, and each ablation
                      vs the baseline (imgsz 640, hsv default) of the same model, and hsv "hue" vs "strong".
                      p_bootstrap: paired image bootstrap (test-set sampling); p_welch_seeds: Welch
                      t-test over the per-seed values (seed variation)

CIs reflect test-image sampling (the mean over seeds is resampled); seed variation is
reported separately as the std over seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["weed", "crop"]
TARGETS = ["greenhouse_test", "field_test"]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
AP_X = np.linspace(0, 1, 101)
CONFIG_KEYS = ["model", "imgsz", "hsv", "epochs"]
BASELINE = {"imgsz": 640, "hsv": "default"}
# Metrics reported in the summaries: (class or "all", AP at IoU 0.5 or averaged over 0.5:0.95)
METRICS = {"weed_AP50": ("weed", "AP50"), "weed_AP50-95": ("weed", "AP50-95"),
           "mAP50": ("all", "AP50"), "mAP50-95": ("all", "AP50-95"), "crop_AP50": ("crop", "AP50")}


def load_ground_truth(data_dir, split):
    """Return image ids and, per image, (classes, xyxy pixel boxes) from the YOLO labels."""
    ids = [Path(line).stem for line in (data_dir / f"{split}.txt").read_text().split()]
    gt = {}
    for i in ids:
        with Image.open(data_dir / "images" / f"{i}.jpg") as im:
            w, h = im.size
        rows = np.loadtxt(data_dir / "labels" / f"{i}.txt", ndmin=2)
        cls = rows[:, 0].astype(int) if len(rows) else np.zeros(0, int)
        cx, cy, bw, bh = (rows[:, 1:].T * np.array([[w], [h], [w], [h]])) if len(rows) else np.zeros((4, 0))
        gt[i] = (cls, np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
    return ids, gt


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match_image(gt_cls, gt_box, det_cls, det_box):
    """Greedy matching in descending confidence order, as Ultralytics' match_predictions."""
    correct = np.zeros((len(det_cls), len(IOU_THRESHOLDS)), bool)
    if len(gt_cls) == 0 or len(det_cls) == 0:
        return correct
    iou = box_iou(gt_box, det_box) * (gt_cls[:, None] == det_cls[None])
    matched = np.zeros((len(gt_cls), len(IOU_THRESHOLDS)), bool)
    cols = np.arange(len(IOU_THRESHOLDS))
    for j in np.flatnonzero((iou >= IOU_THRESHOLDS[0]).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, cols] >= IOU_THRESHOLDS
        matched[k, cols] |= correct[j]
    return correct


class MatchedRun:
    """Matched detections of one run on one test set, for fast weighted (bootstrap) AP."""

    def __init__(self, pred_json, ids, gt):
        preds = pd.DataFrame(json.loads(Path(pred_json).read_text()))
        preds["image_id"] = preds["image_id"].astype(str)
        preds["cls"] = preds["category_id"] - 1  # Ultralytics writes category_id = class + 1
        index = {i: n for n, i in enumerate(ids)}
        img, conf, cls, correct = [], [], [], []
        for image_id, p in preds.groupby("image_id"):
            p = p.sort_values("score", ascending=False)
            xywh = np.array(p["bbox"].tolist())
            det_box = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
            correct.append(match_image(*gt[image_id], p["cls"].values, det_box))
            img.append(np.full(len(p), index[image_id]))
            conf.append(p["score"].values)
            cls.append(p["cls"].values)
        img, conf, cls, correct = map(np.concatenate, (img, conf, cls, correct))
        self.n_gt = np.array([[np.sum(gt[i][0] == c) for c in range(len(CLASSES))] for i in ids])
        self.per_class = {}
        for c in range(len(CLASSES)):
            order = np.flatnonzero(cls == c)[np.argsort(-conf[cls == c], kind="stable")]
            correct_t = np.ascontiguousarray(correct[order].T)  # (thresholds, detections)
            # The interpolated AP only depends on each true positive, the detection just before it, and
            # the last detection; the long runs of false positives in between can be skipped.
            keep = [np.unique(np.concatenate([k, k - 1, [len(order) - 1]]).clip(0)) if len(order) else k
                    for k in (np.flatnonzero(row) for row in correct_t)]
            self.per_class[c] = (img[order], correct_t, keep)

    def ap(self, c, weights):
        """AP per IoU threshold for class c, with per-image weights (all ones = full test set)."""
        n_l = (weights * self.n_gt[:, c]).sum()
        img, correct_t, keep = self.per_class[c]
        if n_l == 0 or len(img) == 0:
            return np.full(len(IOU_THRESHOLDS), np.nan if n_l == 0 else 0.0)
        w = weights[img]
        cum_w = np.cumsum(w)
        out = np.empty(len(IOU_THRESHOLDS))
        for t in range(len(IOU_THRESHOLDS)):
            k = keep[t]
            tpc = np.cumsum(w * correct_t[t])[k]
            recall = tpc / (n_l + 1e-16)
            precision = tpc / np.maximum(cum_w[k], 1e-16)
            mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            out[t] = np.trapezoid(np.interp(AP_X, mrec, mpre), AP_X)
        return out

    def metrics(self, weights):
        ap = {c: self.ap(c, weights) for c in range(len(CLASSES))}
        present = [c for c in ap if not np.isnan(ap[c][0])]  # classes with labels, as Ultralytics
        out = {}
        for name, (cls_name, kind) in METRICS.items():
            if cls_name == "all":
                vals = [ap[c][0] if kind == "AP50" else ap[c].mean() for c in present]
                out[name] = float(np.mean(vals)) if vals else np.nan
            else:
                a = ap[CLASSES.index(cls_name)]
                out[name] = float(a[0] if kind == "AP50" else a.mean())
        return out


def ci(x):
    x = np.asarray(x)[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.runs_dir.iterdir()
                      if (d / "run_config.json").exists() and all((d / f"eval_{t}/predictions.json").exists() for t in TARGETS))
    if not run_dirs:
        raise SystemExit(f"No complete runs in {args.runs_dir}")
    print(f"{len(run_dirs)} runs")

    rng = np.random.default_rng(args.seed)
    gts, weights = {}, {}
    for t in TARGETS:
        ids, gt = load_ground_truth(args.data_dir, t)
        gts[t] = (ids, gt)
        # Multinomial counts = how often each test image appears in a bootstrap resample.
        weights[t] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=args.n_boot).astype(float)

    rows, boot = [], {}  # boot[(run, target)] = DataFrame (n_boot x metrics)
    for d in run_dirs:
        cfg = json.loads((d / "run_config.json").read_text())
        row = {"run": d.name, **{k: cfg[k] for k in CONFIG_KEYS + ["seed"]},
               "train_min": round(cfg.get("train_seconds", np.nan) / 60, 1)}
        reported = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        for t in TARGETS:
            m = MatchedRun(d / f"eval_{t}/predictions.json", *gts[t])
            row.update({f"{t}/{k}": v for k, v in m.metrics(np.ones(len(gts[t][0]))).items()})
            row[f"{t}/mAP50_ultralytics"] = reported.get(t, {}).get("mAP50", np.nan)
            boot[(d.name, t)] = pd.DataFrame([m.metrics(w) for w in weights[t]])
        rows.append(row)
        print(f"  {d.name}: field weed AP50 {row['field_test/weed_AP50']:.3f}")
    runs = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "run_metrics.csv", index=False)

    # Per config: mean over seeds, both for the full test set and for each bootstrap resample.
    configs = {key: g["run"].tolist() for key, g in runs.groupby(CONFIG_KEYS)}

    def config_boot(names, t):
        return sum(boot[(n, t)] for n in names) / len(names)

    summary = []
    for key, names in configs.items():
        g = runs[runs["run"].isin(names)]
        b = {t: config_boot(names, t) for t in TARGETS}
        for metric in METRICS:
            gh, fi = g[f"greenhouse_test/{metric}"], g[f"field_test/{metric}"]
            for label, vals, bvals in [("greenhouse_test", gh, b["greenhouse_test"][metric]),
                                       ("field_test", fi, b["field_test"][metric]),
                                       ("gap", gh - fi, b["greenhouse_test"][metric] - b["field_test"][metric])]:
                lo, hi = ci(bvals)
                summary.append({**dict(zip(CONFIG_KEYS, key)), "n_seeds": len(names), "metric": metric, "target": label,
                                "mean": vals.mean(), "std_seeds": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                                "ci_low": lo, "ci_high": hi})
    pd.DataFrame(summary).to_csv(args.out_dir / "config_summary.csv", index=False)

    # Paired comparisons (same bootstrap resamples on both sides).
    comparisons = []

    def compare(kind, key_a, key_b):
        for metric in METRICS:
            for label in ["field_test", "gap", "greenhouse_test"]:
                def value(key, boot_frame=None):
                    if label == "gap":
                        return config_boot(configs[key], "greenhouse_test")[metric] - config_boot(configs[key], "field_test")[metric]
                    return config_boot(configs[key], label)[metric]

                def point(key):
                    g = runs[runs["run"].isin(configs[key])]
                    if label == "gap":
                        return (g[f"greenhouse_test/{metric}"] - g[f"field_test/{metric}"]).mean()
                    return g[f"{label}/{metric}"].mean()

                def per_seed(key):
                    g = runs[runs["run"].isin(configs[key])]
                    if label == "gap":
                        return (g[f"greenhouse_test/{metric}"] - g[f"field_test/{metric}"]).values
                    return g[f"{label}/{metric}"].values

                diff = (value(key_a) - value(key_b)).values
                lo, hi = ci(diff)
                valid = diff[~np.isnan(diff)]
                p = min(1.0, 2 * min((valid <= 0).mean(), (valid >= 0).mean())) if len(valid) else np.nan
                # Seed level: Welch t-test on the per-run values (test images fixed, seeds vary).
                sa, sb = per_seed(key_a), per_seed(key_b)
                p_seeds = stats.ttest_ind(sa, sb, equal_var=False).pvalue if min(len(sa), len(sb)) >= 2 else np.nan
                comparisons.append({"comparison": kind, "a": "/".join(map(str, key_a)), "b": "/".join(map(str, key_b)),
                                    "metric": metric, "target": label, "diff_a_minus_b": point(key_a) - point(key_b),
                                    "ci_low": lo, "ci_high": hi, "p_bootstrap": p, "p_welch_seeds": p_seeds})

    keys = list(configs)
    for a in keys:
        cfg_a = dict(zip(CONFIG_KEYS, a))
        for b in keys:
            cfg_b = dict(zip(CONFIG_KEYS, b))
            same_settings = all(cfg_a[k] == cfg_b[k] for k in ("imgsz", "hsv", "epochs"))
            if same_settings and cfg_a["model"].startswith("rtdetr") and not cfg_b["model"].startswith("rtdetr"):
                compare("model", a, b)
            is_baseline_b = all(cfg_b[k] == v for k, v in BASELINE.items())
            if a != b and is_baseline_b and cfg_a["model"] == cfg_b["model"] and cfg_a["epochs"] == cfg_b["epochs"]:
                compare("ablation", a, b)
            # "hue" differs from "strong" only in the hue range: isolates the effect of hue invariance.
            same_other = all(cfg_a[k] == cfg_b[k] for k in ("model", "imgsz", "epochs"))
            if same_other and cfg_a["hsv"] == "hue" and cfg_b["hsv"] == "strong":
                compare("hue_vs_strong", a, b)
    pd.DataFrame(comparisons).to_csv(args.out_dir / "comparisons.csv", index=False)

    check = runs[[f"{t}/{k}" for t in TARGETS for k in ("mAP50", "mAP50_ultralytics")]]
    print("\nFull-set mAP50, recomputed vs Ultralytics:")
    print(pd.concat([runs["run"], check], axis=1).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
