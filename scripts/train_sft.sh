#!/usr/bin/env bash
# Deletion Tuning with supervised fine-tuning on one GPU, with the paper's settings. Trains one gate-projection mask per
# task (the 14 MMLU-Pro categories or the 23 BBEH tasks) and saves each to <output_dir>/masks/<task>/.
# METHOD=fft trains the full fine-tuning baseline instead and saves each task's model to
# <output_dir>/models/<task>/.
#
# Usage: [METHOD=fft] bash scripts/train_sft.sh <mmlu_pro|bbeh> <llama1b|llama3b> <output_dir> [extra flags]
# Run from the repository root; BBEH reads its clone at data/bbeh.
set -euo pipefail
BENCHMARK=$1; MODEL=$2; OUT=$3; shift 3

case "$BENCHMARK" in
  mmlu_pro) ARGS=(--train_batch_size 4) ;;
  bbeh) ARGS=(--train_batch_size 2 --grad_accum_steps 2 --data_dir data/bbeh) ;;
  *) echo "unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac
case "${METHOD:-deletion_tuning}" in
  deletion_tuning) ARGS+=(--learning_rate 5e-4) ;;
  fft)
    case "$BENCHMARK" in
      mmlu_pro) ARGS+=(--fullparam --learning_rate 3e-6) ;;
      bbeh) ARGS+=(--fullparam --learning_rate 1e-6) ;;
    esac ;;
  *) echo "METHOD must be deletion_tuning or fft" >&2; exit 1 ;;
esac

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
accelerate launch --num_processes 1 --mixed_precision bf16 src/train_sft.py \
  --benchmark "$BENCHMARK" --model_name "$MODEL" --output_dir "$OUT" \
  --train_steps 400 --init_sparsity 0.00048828125 --sharpness_start 100 --sharpness_end 1000 --weight_reg 1e-5 \
  "${ARGS[@]}" "$@"
