"""Convert the Pascal VOC annotations of all split images to a YOLO dataset.

Writes to data/processed/yolo/:
  images/<id>.jpg, labels/<id>.txt   (class cx cy w h, normalised; 0 = weed, 1 = crop)
  <split>.txt                        image list per split, paths relative to this folder
  greenhouse.yaml                    train/val/test = greenhouse splits
  field.yaml                         val/test = field test set (for evaluation only)

The YAMLs have no `path` key, so Ultralytics resolves the split files relative to the
YAML's own folder and the whole yolo/ folder can be copied anywhere (e.g. Colab).
Degenerate boxes (zero width or height) are dropped and reported. A few converted
images are drawn back from the YOLO labels to results/inspection/yolo_check.jpg.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["weed", "crop"]
CLASS_COLORS = ["#eb6834", "#2a78d6"]
SPLITS = ["greenhouse_train", "greenhouse_val", "greenhouse_test", "field_test"]


def voc_to_yolo(xml_path):
    root = ET.parse(xml_path).getroot()
    w, h = int(root.findtext("size/width")), int(root.findtext("size/height"))
    lines, dropped = [], 0
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        if x1 <= x0 or y1 <= y0:
            dropped += 1
            continue
        cls = CLASSES.index(obj.findtext("name").strip())
        lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return (w, h), lines, dropped


def write_yaml(path, train, val, test):
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    path.write_text(f"train: {train}.txt\nval: {val}.txt\ntest: {test}.txt\nnames:\n{names}\n")


def check_sheet(out_dir, ids, out_path, thumb=360, cols=4):
    rows = int(np.ceil(len(ids) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, image_id in enumerate(ids):
        im = Image.open(out_dir / "images" / f"{image_id}.jpg").convert("RGB")
        d = ImageDraw.Draw(im)
        for line in (out_dir / "labels" / f"{image_id}.txt").read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            cx, bw = float(cx) * im.width, float(bw) * im.width
            cy, bh = float(cy) * im.height, float(bh) * im.height
            d.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=CLASS_COLORS[int(c)], width=max(2, im.width // 250))
        im.thumbnail((thumb, thumb))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/Ronin_OPEN_DB")
    ap.add_argument("--splits-dir", type=Path, default=REPO / "data/processed/splits")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/processed/yolo")
    ap.add_argument("--check-image", type=Path, default=REPO / "results/inspection/yolo_check.jpg")
    args = ap.parse_args()

    img_dir = args.raw_dir / "raw images"
    raw_images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels").mkdir(parents=True, exist_ok=True)

    split_ids = {s: (args.splits_dir / f"{s}.txt").read_text().split() for s in SPLITS}
    n_boxes, n_dropped = 0, 0
    for image_id in sorted({i for ids in split_ids.values() for i in ids}):
        src = raw_images[image_id]
        (w, h), lines, dropped = voc_to_yolo(args.raw_dir / "annotations" / f"{image_id}.xml")
        with Image.open(src) as im:
            assert im.size == (w, h), f"{image_id}: XML size {(w, h)} != image size {im.size}"
        if dropped:
            print(f"{image_id}: dropped {dropped} degenerate box(es)")
        n_boxes, n_dropped = n_boxes + len(lines), n_dropped + dropped
        shutil.copyfile(src, args.out_dir / "images" / f"{image_id}.jpg")
        (args.out_dir / "labels" / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    for s, ids in split_ids.items():
        (args.out_dir / f"{s}.txt").write_text("".join(f"./images/{i}.jpg\n" for i in ids))
    write_yaml(args.out_dir / "greenhouse.yaml", "greenhouse_train", "greenhouse_val", "greenhouse_test")
    write_yaml(args.out_dir / "field.yaml", "greenhouse_train", "field_test", "field_test")

    rng = np.random.default_rng(0)
    sample = [*rng.choice(split_ids["greenhouse_train"], 4, replace=False), *rng.choice(split_ids["field_test"], 4, replace=False)]
    args.check_image.parent.mkdir(parents=True, exist_ok=True)
    check_sheet(args.out_dir, sample, args.check_image)
    print(f"{sum(map(len, split_ids.values()))} images, {n_boxes} boxes written, {n_dropped} dropped -> {args.out_dir}")


if __name__ == "__main__":
    main()
