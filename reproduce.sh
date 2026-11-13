#!/usr/bin/env bash
# Reproduce one training run and its evaluation end to end.
#
#   bash reproduce.sh            Full data (unzipped into data/raw/, see README): domains, splits, conversion,
#                                one YOLO11s baseline run (seed 0, 100 epochs; needs a GPU, ~6 min on a T4),
#                                bootstrap analysis and error analysis -> runs/reproduce/, results/reproduce/
#   bash reproduce.sh --sample   Bundled sample_data/ (40 images), 3 epochs on CPU (a few minutes): checks that
#                                the pipeline runs; the numbers are meaningless -> runs/reproduce_sample/, results/reproduce_sample/
#
# Set PYTHON to choose the interpreter (default: python), e.g. PYTHON=python3 bash reproduce.sh --sample
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

if [[ "${1:-}" == "--sample" ]]; then
    RAW=sample_data/Ronin_OPEN_DB
    SPLITS=sample_data/splits          # the fixed splits, restricted to the sample
    DATA=data/sample/yolo
    RUNS=runs/reproduce_sample
    OUT=results/reproduce_sample
    TRAIN_ARGS=(--epochs 3 --batch 8 --device cpu --workers 0)
else
    RAW=data/raw/Ronin_OPEN_DB
    SPLITS=data/processed/splits
    DATA=data/processed/yolo
    RUNS=runs/reproduce
    OUT=results/reproduce
    TRAIN_ARGS=(--epochs 100 --batch 16)
    # Regenerate domains and splits from the raw data (deterministic: seed 0, identical to the committed ones).
    "$PY" src/assign_domains.py --raw-dir "$RAW"
    "$PY" src/make_splits.py --raw-dir "$RAW" --out-dir "$SPLITS" --summary "$OUT/splits_summary.csv"
fi
mkdir -p "$OUT"

"$PY" src/convert_to_yolo.py --raw-dir "$RAW" --splits-dir "$SPLITS" --out-dir "$DATA" --check-image "$OUT/yolo_check.jpg"
"$PY" src/train.py --model yolo11s --seed 0 --data-dir "$DATA" --project "$RUNS" "${TRAIN_ARGS[@]}"
"$PY" src/analyze.py --runs-dir "$RUNS" --data-dir "$DATA" --out-dir "$OUT"
"$PY" src/error_analysis.py --runs-dir "$RUNS" --data-dir "$DATA" --out-dir "$OUT"
echo "Done: run in $RUNS/, metrics in $OUT/run_metrics.csv and $OUT/errors_per_run.csv"
