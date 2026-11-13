"""Train one detector on the greenhouse split and evaluate it in- and cross-domain.

One call = one run. The run folder <project>/<name>/ gets:
  weights/best.pt, results.csv, args.yaml     (Ultralytics; best.pt chosen on greenhouse val)
  run_config.json                             our settings, versions, git commit, split hashes
  metrics.json                                greenhouse test + field test metrics (overall and per class)
  eval_<target>/predictions.json              per-image detections (COCO format) for bootstrap and error analysis

The field test set is only touched after training, in the final evaluation.

Example:
  python src/train.py --model yolo11s --data-dir data/processed/yolo --seed 0 --project runs
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR, YOLO

REPO = Path(__file__).resolve().parents[1]
# Colour-augmentation settings for the HSV ablation; "default" is Ultralytics' default.
HSV = {
    "default": dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4),
    "off": dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0),
    "strong": dict(hsv_h=0.05, hsv_s=0.9, hsv_v=0.6),
    # As "strong" but hue shifted by up to +-0.5 * 180, i.e. anywhere on the hue circle: tests whether hue
    # invariance helps (field vs greenhouse weed hue differs by ~55 of 180, see src/colour_stats.py).
    "hue": dict(hsv_h=0.5, hsv_s=0.9, hsv_v=0.6),
}
EVAL_TARGETS = {"greenhouse_test": "greenhouse.yaml", "field_test": "field.yaml"}


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def metrics_dict(m, names):
    per_class = {
        names[int(c)]: {"AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i]),
                        "P": float(m.box.p[i]), "R": float(m.box.r[i])}
        for i, c in enumerate(m.box.ap_class_index)
    }
    return {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "P": float(m.box.mp), "R": float(m.box.mr),
            "per_class": per_class}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. yolo11s or rtdetr-l (COCO-pretrained weights)")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--project", type=Path, default=REPO / "runs")
    ap.add_argument("--name", help="run folder name (default: <model>_img<imgsz>_hsv-<hsv>_s<seed>)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--hsv", choices=HSV, default="default")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    name = args.name or f"{args.model}_img{args.imgsz}_hsv-{args.hsv}_s{args.seed}"
    run_dir = args.project / name
    data_dir = args.data_dir.resolve()
    model = (RTDETR if args.model.startswith("rtdetr") else YOLO)(f"{args.model}.pt")

    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "name": name, "hsv_values": HSV[args.hsv], "git_commit": git_commit(),
        "split_hashes": {f.name: file_hash(f) for f in sorted(data_dir.glob("*.txt"))},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    t0 = time.time()
    model.train(
        data=str(data_dir / "greenhouse.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, device=args.device, workers=args.workers,
        patience=args.epochs,  # fixed number of epochs: no early stopping
        project=str(args.project.resolve()), name=name, exist_ok=True, **HSV[args.hsv],
    )
    config["train_seconds"] = round(time.time() - t0)
    if torch.cuda.is_available():
        config["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    best = (RTDETR if args.model.startswith("rtdetr") else YOLO)(str(run_dir / "weights/best.pt"))
    metrics = {}
    for target, yaml_name in EVAL_TARGETS.items():
        m = best.val(data=str(data_dir / yaml_name), split="test", imgsz=args.imgsz, batch=args.batch,
                     device=args.device, save_json=True, project=str(run_dir.resolve()), name=f"eval_{target}",
                     exist_ok=True)
        metrics[target] = metrics_dict(m, best.names)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({t: {k: round(v, 4) for k, v in m.items() if k != "per_class"} for t, m in metrics.items()}))


if __name__ == "__main__":
    main()
