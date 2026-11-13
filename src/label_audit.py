"""Manual audit of field "background" false positives: unlabelled plant or real error?

  make       Draw a uniform random sample of background detections (conf >= --conf, IoU < 0.1 with
             every ground-truth box) per run, mix them in random order (model hidden) and write a
             self-contained HTML page. Open it in a browser, answer with keys 1-4, export the CSV.
  summarise  Read the exported CSV and report, per model, the share of each answer with a 95%
             Wilson interval -> results/label_audit/summary.csv

Examples:
  python src/label_audit.py make --runs yolo11s_img640_hsv-default_s0 rtdetr-l_img640_hsv-default_s0
  python src/label_audit.py summarise --answers results/label_audit/answers.csv
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze import load_ground_truth
from error_analysis import CLASS_COLORS, ERROR_COLOR, analyse_run

REPO = Path(__file__).resolve().parents[1]
ANSWERS = {"1": "unlabelled weed", "2": "unlabelled crop plant", "3": "real background", "4": "unsure"}


def crop_jpeg(im, box, half, size):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    c = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((size, size))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make(args):
    ids, gt = load_ground_truth(args.data_dir, "field_test")
    sizes = {}
    for i in ids:
        with Image.open(args.data_dir / "images" / f"{i}.jpg") as im:
            sizes[i] = im.size
    rng = np.random.default_rng(args.seed)
    items = []
    for run in args.runs:
        cfg = json.loads((args.runs_dir / run / "run_config.json").read_text())
        rec = analyse_run(args.runs_dir / run, "field_test", ids, gt, sizes, args.conf, cfg["imgsz"])
        bg = rec[rec["type"] == "background"]
        sample = bg.iloc[rng.choice(len(bg), min(args.n, len(bg)), replace=False)]
        print(f"{run}: {len(bg)} background detections at conf >= {args.conf}, sampled {len(sample)}")
        for _, r in sample.iterrows():
            items.append({"run": run, "model": cfg["model"], "image_id": r["image_id"], "cls": r["cls"],
                          "score": round(float(r["score"]), 3), "box": r["box"], "n_background": len(bg)})
    order = rng.permutation(len(items))
    items = [items[k] for k in order]

    for n, it in enumerate(items):
        with Image.open(args.data_dir / "images" / f"{it['image_id']}.jpg") as im:
            im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        for c, b in zip(*gt[it["image_id"]]):
            d.rectangle(b.tolist(), outline=CLASS_COLORS[c], width=2)
        x0, y0, x1, y1 = it["box"]
        d.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=ERROR_COLOR, width=3)
        side = max(x1 - x0, y1 - y0)
        it["close"] = crop_jpeg(im, it["box"], max(40, 1.5 * side), 320)
        it["context"] = crop_jpeg(im, it["box"], max(160, 5 * side), 320)
        it["id"] = n

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__ITEMS__", json.dumps(items)).replace("__ANSWERS__", json.dumps(ANSWERS)) \
                   .replace("__KEY__", f"label-audit-{args.seed}-{len(items)}")
    out = args.out_dir / "audit.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(items)} items -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarise(args):
    a = pd.read_csv(args.answers)
    a = a[a["answer"].notna() & (a["answer"] != "")]
    rows = []
    for model, g in a.groupby("model"):
        for answer in ANSWERS.values():
            k, n = int((g["answer"] == answer).sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"model": model, "answer": answer, "count": k, "n": n, "share": k / n,
                         "ci_low": lo, "ci_high": hi, "n_background_total": int(g["n_background"].iloc[0])})
    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "summary.csv", index=False)
    print(out.round(3).to_string(index=False))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label audit</title>
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e5e4df; --accent:#2a78d6; --chip:#f1f0ec; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3936; --accent:#3987e5; --chip:#2a2a28; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
main { max-width:720px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 16px; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:8px 0 16px; }
.bar > div { height:100%; background:var(--accent); width:0; }
.imgs { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.imgs figure { margin:0; } .imgs img { width:100%; display:block; border-radius:6px; }
.imgs figcaption { color:var(--muted); font-size:13px; margin-top:2px; }
.meta { color:var(--muted); font-size:13px; margin:10px 0; }
.btns { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
button { font:inherit; padding:10px 12px; border-radius:8px; border:1px solid var(--line); background:var(--chip); color:var(--ink); cursor:pointer; text-align:left; }
button.sel { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent) inset; }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-radius:4px; padding:0 5px; margin-right:6px; }
.nav { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; } .nav button { text-align:center; }
.legend span { display:inline-block; width:12px; height:3px; vertical-align:middle; margin:0 4px 0 10px; }
</style></head><body><main>
<h1>Label audit: field "background" detections</h1>
<p class="sub">The <b>yellow box</b> is a detection that overlaps no labelled object. What is inside it?
<span class="legend">Labels: <span style="background:#eb6834"></span>weed <span style="background:#2a78d6"></span>crop</span></p>
<div id="progress"></div><div class="bar"><div id="fill"></div></div>
<div class="imgs"><figure><img id="close" alt="close-up"><figcaption>close-up</figcaption></figure>
<figure><img id="context" alt="context"><figcaption>context</figcaption></figure></div>
<div class="meta" id="meta"></div>
<div class="btns" id="btns"></div>
<div class="nav"><button id="prev"><kbd>&larr;</kbd>Previous</button><button id="next">Next<kbd style="margin:0 0 0 6px">&rarr;</kbd></button>
<button id="export">Export CSV</button></div>
<p class="sub" style="margin-top:16px">Answers are saved in this browser as you go. When done, click <b>Export CSV</b> and save the file as <code>results/label_audit/answers.csv</code>.</p>
</main><script>
const ITEMS = __ITEMS__, ANSWERS = __ANSWERS__, KEY = "__KEY__";
let answers = {}; try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let pos = Math.min(Object.keys(answers).length, ITEMS.length - 1);
const $ = id => document.getElementById(id);
for (const [k, v] of Object.entries(ANSWERS)) {
  const b = document.createElement("button"); b.dataset.key = k; b.innerHTML = `<kbd>${k}</kbd>${v}`;
  b.onclick = () => answer(k); $("btns").appendChild(b);
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} }
function show() {
  const it = ITEMS[pos], done = Object.keys(answers).length;
  $("close").src = it.close; $("context").src = it.context;
  $("meta").textContent = `Item ${pos + 1} of ${ITEMS.length} · predicted "${it.cls}", confidence ${it.score}`;
  $("progress").textContent = `${done} of ${ITEMS.length} answered`;
  $("fill").style.width = (100 * done / ITEMS.length) + "%";
  for (const b of $("btns").children) b.classList.toggle("sel", answers[it.id] === ANSWERS[b.dataset.key]);
}
function answer(k) { answers[ITEMS[pos].id] = ANSWERS[k]; save(); if (pos < ITEMS.length - 1) pos++; show(); }
$("prev").onclick = () => { if (pos > 0) pos--; show(); };
$("next").onclick = () => { if (pos < ITEMS.length - 1) pos++; show(); };
document.addEventListener("keydown", e => {
  if (ANSWERS[e.key]) answer(e.key);
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
});
$("export").onclick = () => {
  const cols = ["id", "run", "model", "image_id", "cls", "score", "box", "n_background", "answer"];
  const esc = v => `"${String(v).replace(/"/g, '""')}"`;
  const lines = [cols.join(",")].concat(ITEMS.map(it => cols.map(c =>
    esc(c === "answer" ? (answers[it.id] || "") : c === "box" ? it.box.join(" ") : it[c])).join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], {type: "text/csv"}));
  a.download = "answers.csv"; a.click();
};
show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--runs", nargs="+", required=True, help="run folder names (e.g. the seed-0 baselines)")
    m.add_argument("--n", type=int, default=100, help="detections sampled per run")
    m.add_argument("--conf", type=float, default=0.25)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarise")
    s.add_argument("--answers", type=Path, default=REPO / "results/label_audit/answers.csv")
    for p in (m, s):
        p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
        p.add_argument("--data-dir", type=Path, default=REPO / "data/processed/yolo")
        p.add_argument("--out-dir", type=Path, default=REPO / "results/label_audit")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else summarise(args)


if __name__ == "__main__":
    main()
