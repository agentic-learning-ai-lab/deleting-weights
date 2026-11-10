#!/usr/bin/env bash
# Train a Deletion Tuning mask with on-policy self-distillation on ToolAlpaca (tooluse) or Science, with
# Qwen2.5-7B-Instruct on one GPU and the paper's hyperparameters. The mask is saved every 100 steps on ToolAlpaca, every
# 25 on Science, and at the end. Extra arguments go to sdft/train.py.
# METHOD=fft runs the full fine-tuning baseline instead (sdft/train_fullparam.py with the authors' settings: learning
# rate 5e-5, 2 epochs) and saves the weights at the same steps to <output_dir>/checkpoint-<N>/.
#
# Usage: [METHOD=fft] bash scripts/train_sdft.sh <tooluse|science> <output_dir> [flags]
# Run from the repository root after bash sdft/setup.sh.
set -euo pipefail
DATASET=$1; OUT=$(realpath -m "$2"); shift 2
case $DATASET in
  tooluse) LR=5e-3; SHARPNESS_END=10000; SAVE_STEPS=100 ;;
  science) LR=3e-3; SHARPNESS_END=1000; SAVE_STEPS=25 ;;
  *) echo "dataset must be tooluse or science" >&2; exit 1 ;;
esac
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
cd sdft
if [ "${METHOD:-deletion_tuning}" = fft ]; then
  exec python train_fullparam.py --dataset_name "$DATASET" --model_name Qwen/Qwen2.5-7B-Instruct --output_dir "$OUT" \
    --learning_rate 5e-5 --num_train_epochs 2 --save_steps "$SAVE_STEPS" "$@"
fi
exec python train.py \
  --dataset_name "$DATASET" --model_name Qwen/Qwen2.5-7B-Instruct --output_dir "$OUT" \
  --learning_rate "$LR" --sharpness_end "$SHARPNESS_END" --save_steps "$SAVE_STEPS" "$@"
