# Weed Detection under Domain Shift

Does a weed detector trained on greenhouse images still work on field images, and if not, why? This project trains a CNN detector (YOLO11s) and a transformer detector (RT-DETR-l) on the greenhouse images of the Ronin open weed dataset and tests them, without adaptation, on 908 field images. Assignment 3 for Key Topics in AI (Open Universiteit).

## Main results

Weed AP50, mean over 3 seeds (full tables in `results/config_summary.csv` and `results/comparisons.csv`):

| | YOLO11s greenhouse | YOLO11s field | RT-DETR-l greenhouse | RT-DETR-l field |
|---|---|---|---|---|
| baseline (640 px) | 0.729 | 0.485 | 0.739 | 0.433 |
| 1024 px | 0.826 | 0.437 | 0.853 | 0.410 |
| HSV augmentation off | 0.723 | 0.471 | 0.773 | 0.471 |
| HSV augmentation strong | 0.732 | 0.512 | 0.765 | 0.505 |
| full hue randomisation | 0.667 | 0.242 | 0.743 | 0.281 |

- Both detectors lose a third or more of their weed AP50 in the field. The transformer does not transfer better.
- A higher resolution helps in the greenhouse but widens the domain gap for both models.
- Stronger colour augmentation gives the best field scores. Full hue randomisation halves them.
- A blind audit of 100 field "false positives" per model found that most are weeds without a label (87% for YOLO, 56% for RT-DETR).
- With three seeds, the difference between the two models is not statistically reliable.

## Layout

```
data/raw/         Original Mendeley download, unmodified (git-ignored)
data/processed/   YOLO-format conversion + saved split files (git-ignored)
src/              Inspection, conversion, training and evaluation scripts
notebooks/        Kaggle notebook that ran the training jobs
runs/             Training outputs: weights, metrics (git-ignored)
results/          CSVs, plots and tables used in the report, error examples, label audit
sample_data/      40 images for a quick end-to-end check (see below)
report/           Final report PDF
```

## Data

Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/nj4vtk4tt6/1) and unzip it into `data/raw/` so that the result is
`data/raw/Ronin_OPEN_DB/{annotations,raw images}/`.

Dataset: Sudars, K., Jasko, J., Namatevs, I., Ozola, L., & Badaukis, N. (2020). Dataset of annotated food crops and weed images for robotic computer vision control. *Data in Brief*, 31, 105833. https://doi.org/10.1016/j.dib.2020.105833. Data: https://doi.org/10.17632/nj4vtk4tt6.1, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The images in `results/inspection/` are derived from this dataset.

`sample_data/` holds 40 images from this dataset (16 greenhouse train, 4 val, 4 test, 16 field test; made by `src/make_sample.py`) in the raw layout, with the fixed splits restricted to them. Same source and licence as above.

## Reproduce one run

```bash
bash reproduce.sh --sample   # bundled sample, 3 epochs on CPU, ~2 min: checks the whole pipeline runs
bash reproduce.sh            # full data in data/raw/: one YOLO11s baseline run (seed 0, 100 epochs, GPU) + evaluation
```

Both run conversion → training → evaluation on greenhouse test and field test → bootstrap analysis → error analysis, and write to `runs/reproduce*/` and `results/reproduce*/`. Set `PYTHON=...` to choose the interpreter.

## Pipeline

```bash
pip install -r requirements.txt

python src/inspect_data.py        # stats + contact sheets -> results/inspection/
python src/assign_domains.py      # image -> greenhouse / field / uncertain
python src/make_splits.py         # fixed splits (seed 0) -> data/processed/splits/
python src/convert_to_yolo.py     # VOC -> YOLO dataset -> data/processed/yolo/

# one training run + evaluation on greenhouse test and field test
python src/train.py --model yolo11s --seed 0 --project runs

# all runs -> results/run_metrics.csv, config_summary.csv, comparisons.csv (bootstrap CIs)
python src/analyze.py --runs-dir runs

# error types per run (missed, background FP, crop<->weed, localisation) + example crops
python src/error_analysis.py --runs-dir runs

# blind label audit of field background FPs: make the page, answer it in a browser, then summarise
python src/label_audit.py make --runs yolo11s_img640_hsv-default_s0 rtdetr-l_img640_hsv-default_s0
python src/label_audit.py summarise   # answers.csv -> results/label_audit/summary.csv

# report figures -> results/figures/ (PDF + PNG), LaTeX tables -> report/tables/ (local only)
python src/make_figures.py
python src/make_tables.py

# greenhouse vs field colour (HSV) distributions -> results/colour_*.csv, results/figures/colour_hist
python src/colour_stats.py

# two explanatory figures (domains, IoU) -> results/figures/guide_*.png
python src/make_guide_figures.py
```

Report: `report/report.pdf` (the final submitted version); the LaTeX source is not published.

Training needs a GPU. `notebooks/kaggle_train.ipynb` runs a list of runs on Kaggle; `python src/package_for_kaggle.py --code` makes the zips to upload as Kaggle datasets.

## Development notes

The source code in this repository was developed with assistance from Claude (Anthropic), an AI assistant, used primarily for code generation, debugging, and architectural decisions. All experimental results, analysis, and written conclusions are the author's own.
