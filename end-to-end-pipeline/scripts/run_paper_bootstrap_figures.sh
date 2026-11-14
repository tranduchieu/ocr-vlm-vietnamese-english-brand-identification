#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/tinnl/Giaitri/AI/Projects/Hieu"
E2E="$ROOT/end-to-end-pipeline"
VENV="$E2E/venv/bin/python3"
GT="$ROOT/paper_dataset_release/brand_dataset/benchmark/benchmark_gt_scoring.csv"
PRED="$E2E/outputs/paper_rerun_20261113/predictions"
BOOT="$E2E/outputs/paper_rerun_20261113/bootstrap"
BS="$E2E/experiments/brand_verification_benchmark/scripts/bootstrap_ci.py"
mkdir -p "$BOOT"

run_pair() {
  local name="$1" a="$2" b="$3"
  echo "bootstrap $name ..."
  "$VENV" "$BS" --gt "$GT" --predictions-a "$PRED/$a" --predictions-b "$PRED/$b" \
    --resamples 10000 --seed 42 > "$BOOT/${name}.json"
}

run_pair L0_vs_B0 b0.jsonl l0.jsonl
run_pair B1_vs_B0 b0.jsonl b1.jsonl
run_pair M0_vs_L0 l0.jsonl m0.jsonl
run_pair M3_vs_M0 m0.jsonl m3.jsonl

"$VENV" "$ROOT/paper_figures/gen_data_figures.py"
echo "figures OK"
