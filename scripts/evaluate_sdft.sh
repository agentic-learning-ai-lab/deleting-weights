#!/usr/bin/env bash
# Evaluate a Deletion Tuning mask, a full fine-tuning checkpoint (a model directory), or the base model (with "base")
# on the ToolAlpaca (tooluse) or Science test set and print its accuracy.
#
# Usage: bash scripts/evaluate_sdft.sh <tooluse|science> <mask_dir|checkpoint_dir|base> <output_dir>
# Run from the repository root after bash sdft/setup.sh.
set -euo pipefail
DATASET=$1; CHECKPOINT=$2; OUT=$3

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python sdft/evaluate.py --dataset_name "$DATASET" --model_name Qwen/Qwen2.5-7B-Instruct --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT"
