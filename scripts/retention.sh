#!/usr/bin/env bash
# Forgetting of a Llama GSM8K Deletion Tuning run: prior-task accuracy and forward KL from the base model
# at steps 100, 200, and 348, plus the base model's prior-task accuracy.
#
# Usage: bash scripts/retention.sh <llama1b|llama3b|llama8b> <run_dir> <output_dir>
set -euo pipefail
MODEL=$1; RUN=$2; OUT=$3
case "$MODEL" in
  llama1b) BASE=meta-llama/Llama-3.2-1B-Instruct ;;
  llama3b) BASE=meta-llama/Llama-3.2-3B-Instruct ;;
  llama8b) BASE=meta-llama/Meta-Llama-3-8B-Instruct ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python src/retention_prior_tasks.py --model_name "$BASE" --checkpoint base --output_dir "$OUT/prior_tasks_base"
for STEP in 100 200 348; do
  MASK="$RUN/global_step_$STEP/mask"
  python src/retention_prior_tasks.py --model_name "$BASE" --checkpoint "$MASK" \
    --output_dir "$OUT/prior_tasks_step$STEP"
  python src/retention_forward_kl.py --model_name "$BASE" --mask_dir "$MASK" --val_parquets data/gsm8k/test.parquet \
    --output_dir "$OUT/forward_kl_step$STEP"
done
