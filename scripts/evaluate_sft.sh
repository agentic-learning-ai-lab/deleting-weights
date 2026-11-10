#!/usr/bin/env bash
# Evaluate a run's SFT masks (<output_dir>/masks/) or full fine-tuned models (<output_dir>/models/), one per task,
# on each task's test split and print the accuracy per task and the mean over tasks.
#
# Usage: bash scripts/evaluate_sft.sh <mmlu_pro|bbeh> <llama1b|llama3b> <masks_or_models_dir> <output_dir> [extra flags]
# Run from the repository root; BBEH reads its clone at data/bbeh.
set -euo pipefail
BENCHMARK=$1; MODEL=$2; TRAINED=$3; OUT=$4; shift 4

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python src/evaluate_sft.py --benchmark "$BENCHMARK" --model_name "$MODEL" --trained_dir "$TRAINED" \
  --data_dir data/bbeh --output_dir "$OUT" "$@"
