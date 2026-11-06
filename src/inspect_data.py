"""Day-1 inspection of the raw Ronin dataset.

Reads data/raw/ (read-only) and writes to results/inspection/:
  per_image.csv      one row per image: group, EXIF camera, sizes, box counts, checks
  per_box.csv        one row per bounding box
  group_summary.csv  counts and box-size statistics per group
  contact_<group>.jpg  random sample of images per group with boxes drawn
  box_sizes.png      box size distributions per group and class

Groups are assigned from the filename pattern only (numeric vs IMG_*); whether a
group is greenhouse or field is decided afterwards, from these outputs.
"""

import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[1]
CLASS_COLORS = {"crop": "#2a78d6", "weed": "#eb6834"}  # dataviz palette slots 1-2
GROUP_COLORS = {"numeric": "#2a78d6", "IMG": "#eb6834"}
EXIF_ORIENTATION = 0x0112


def name_group(filename):
    if re.fullmatch(r"\d+\.jpe?g", filename, re.IGNORECASE):
        return "numeric"
    if re.fullmatch(r"IMG_\d+\.jpe?g", filename, re.IGNORECASE):
        return "IMG"
    return "other"


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        boxes.append(
            {
                "cls": obj.findtext("name").strip(),
                **{k: float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")},
            }
        )
    return {
        "xml_filename": root.findtext("filename"),
        "xml_w": int(size.findtext("width")),
        "xml_h": int(size.findtext("height")),
        "boxes": boxes,
    }


def read_exif(img):
    exif = img.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    return {
        "exif_make": str(tags.get("Make", "")).strip("\x00 "),
        "exif_model": str(tags.get("Model", "")).strip("\x00 "),
        "exif_datetime": str(tags.get("DateTime", "")),
        "exif_orientation": exif.get(EXIF_ORIENTATION, ""),
    }


def hsv_means(img):
    small = img.convert("RGB").resize((128, 128)).convert("HSV")
    h, s, v = np.asarray(small, dtype=float).reshape(-1, 3).T
    return {"mean_h": h.mean(), "mean_s": s.mean(), "mean_v": v.mean()}


def collect(raw_dir):
    ann_dir, img_dir = raw_dir / "annotations", raw_dir / "raw images"
    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    xmls = {p.stem: p for p in ann_dir.glob("*.xml")}
    unmatched = sorted(set(images) ^ set(xmls))
    if unmatched:
        print(f"WARNING: {len(unmatched)} images/annotations without a partner: {unmatched[:10]}")

    img_rows, box_rows = [], []
    for stem in sorted(set(images) & set(xmls)):
        img_path = images[stem]
        ann = parse_xml(xmls[stem])
        with Image.open(img_path) as img:
            raw_w, raw_h = img.size
            row = {
                "image_id": stem,
                "file": img_path.name,
                "group": name_group(img_path.name),
                "raw_w": raw_w,
                "raw_h": raw_h,
                **read_exif(img),
                **hsv_means(img),
            }
        n_out_of_bounds = sum(
            b["xmin"] < 0 or b["ymin"] < 0 or b["xmax"] > ann["xml_w"] or b["ymax"] > ann["xml_h"]
            for b in ann["boxes"]
        )
        row.update(
            xml_filename=ann["xml_filename"],
            xml_w=ann["xml_w"],
            xml_h=ann["xml_h"],
            filename_match=ann["xml_filename"] == img_path.name,
            size_match=(ann["xml_w"], ann["xml_h"]) == (raw_w, raw_h),
            size_match_transposed=(ann["xml_w"], ann["xml_h"]) == (raw_h, raw_w),
            n_weed=sum(b["cls"] == "weed" for b in ann["boxes"]),
            n_crop=sum(b["cls"] == "crop" for b in ann["boxes"]),
            n_other=sum(b["cls"] not in ("weed", "crop") for b in ann["boxes"]),
            n_degenerate=sum(b["xmax"] <= b["xmin"] or b["ymax"] <= b["ymin"] for b in ann["boxes"]),
            n_out_of_bounds=n_out_of_bounds,
        )
        img_rows.append(row)
        for b in ann["boxes"]:
            w, h = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
            box_rows.append(
                {
                    "image_id": stem,
                    "group": row["group"],
                    **b,
                    "w": w,
                    "h": h,
                    # sqrt of box area as a fraction of image area: side length relative to the image
                    "rel_size": np.sqrt(max(w, 0) * max(h, 0) / (ann["xml_w"] * ann["xml_h"])),
                    # box side length after letterboxing the image to 640 px (YOLO default imgsz)
                    "side_at_640": np.sqrt(max(w, 0) * max(h, 0)) * 640 / max(ann["xml_w"], ann["xml_h"]),
                }
            )
    return pd.DataFrame(img_rows), pd.DataFrame(box_rows)


def summarise(imgs, boxes):
    rows = []
    for group, g in imgs.groupby("group"):
        b = boxes[boxes["group"] == group]
        cameras = g.groupby(["exif_make", "exif_model"]).size()
        sizes = g.groupby(["raw_w", "raw_h"]).size().sort_values(ascending=False)
        rows.append(
            {
                "group": group,
                "n_images": len(g),
                "n_images_with_crop": int((g["n_crop"] > 0).sum()),
                "n_images_without_boxes": int((g["n_weed"] + g["n_crop"] + g["n_other"] == 0).sum()),
                "n_weed_boxes": int(g["n_weed"].sum()),
                "n_crop_boxes": int(g["n_crop"].sum()),
                "boxes_per_image_mean": (g["n_weed"] + g["n_crop"]).mean(),
                "cameras": "; ".join(f"{(mk + ' ' + md).strip() or '(no EXIF)'}: {n}" for (mk, md), n in cameras.items()),
                "image_sizes": "; ".join(f"{w}x{h}: {n}" for (w, h), n in sizes.items()),
                "n_size_mismatch": int((~g["size_match"]).sum()),
                "n_exif_rotated": int(g["exif_orientation"].isin([5, 6, 7, 8]).sum()),
                "rel_size_median_weed": b.loc[b["cls"] == "weed", "rel_size"].median(),
                "rel_size_median_crop": b.loc[b["cls"] == "crop", "rel_size"].median(),
                "side_at_640_p10": b["side_at_640"].quantile(0.1),
                "side_at_640_median": b["side_at_640"].median(),
                "n_boxes_side_lt_16px_at_640": int((b["side_at_640"] < 16).sum()),
                "mean_h": g["mean_h"].mean(),
                "mean_s": g["mean_s"].mean(),
                "mean_v": g["mean_v"].mean(),
            }
        )
    return pd.DataFrame(rows)


def contact_sheet(imgs, boxes, raw_dir, group, out_path, n=48, cols=8, thumb=240, seed=0):
    sample = imgs[imgs["group"] == group].sample(min(n, (imgs["group"] == group).sum()), random_state=seed)
    rows = int(np.ceil(len(sample) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    for i, (_, r) in enumerate(sample.iterrows()):
        with Image.open(raw_dir / "raw images" / r["file"]) as im:
            # Draw boxes on the un-rotated pixels: XML coordinates refer to the stored pixel grid
            # when size_match is True. Rotation is checked separately (exif_orientation column).
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            lw = max(2, im.width // 200)
            for _, b in boxes[boxes["image_id"] == r["image_id"]].iterrows():
                d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=CLASS_COLORS.get(b["cls"], "white"), width=lw)
            im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
        sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw_sheet.text((x + 4, y + thumb + 2), f"{r['file']} {r['raw_w']}x{r['raw_h']}", fill="#52514e")
    sheet.save(out_path, quality=90)


def plot_box_sizes(boxes, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    bins = np.logspace(np.log10(2), np.log10(640), 40)
    for ax, cls in zip(axes, ["weed", "crop"]):
        for group, color in GROUP_COLORS.items():
            vals = boxes.loc[(boxes["cls"] == cls) & (boxes["group"] == group), "side_at_640"]
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=color, label=f"{group} (n={len(vals)})")
        ax.axvline(16, color="#52514e", linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(f"{cls} boxes", loc="left", fontsize=11)
        ax.set_xlabel("box side length at imgsz 640 (px, sqrt of area)")
        ax.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("number of boxes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/Ronin_OPEN_DB")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/inspection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    imgs, boxes = collect(args.raw_dir)
    summary = summarise(imgs, boxes)
    imgs.to_csv(args.out_dir / "per_image.csv", index=False)
    boxes.to_csv(args.out_dir / "per_box.csv", index=False)
    summary.to_csv(args.out_dir / "group_summary.csv", index=False)

    for group in sorted(imgs["group"].unique()):
        contact_sheet(imgs, boxes, args.raw_dir, group, args.out_dir / f"contact_{group}.jpg", seed=args.seed)
    plot_box_sizes(boxes, args.out_dir / "box_sizes.png")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.T.to_string())


if __name__ == "__main__":
    main()
