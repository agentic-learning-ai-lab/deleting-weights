#!/usr/bin/env bash
# Build the GRPO training and validation parquets under data/ (GSM8K and SimpleRL-Zoo).
# Run from the repository root with the GRPO environment active.
set -euo pipefail

# GSM8K (openai/gsm8k): data/gsm8k/{train,test}.parquet
python verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k

# SimpleRL-Zoo training data: data/simplerl_zoo/simplelr_{abel_level1to4,qwen_level3to5}/train.parquet
hf download hkust-nlp/SimpleRL-Zoo-Data --repo-type dataset \
    --revision 50ec8f09045e785fd20c13a508820e77cccb4211 --local-dir data/raw/SimpleRL-Zoo-Data
python src/prepare_simplerl_zoo_train.py --src data/raw/SimpleRL-Zoo-Data --dst data/simplerl_zoo

# SimpleRL-Zoo validation benchmarks from simpleRL-reason: data/simplerl_zoo/eval/{qwen,abel}/*.parquet
if [ ! -d data/raw/simpleRL-reason ]; then
    git clone --filter=blob:none --no-checkout https://github.com/hkust-nlp/simpleRL-reason.git data/raw/simpleRL-reason
    git -C data/raw/simpleRL-reason sparse-checkout set examples/simplelr_math_eval/data
    git -C data/raw/simpleRL-reason checkout cf1c7858bfc145a27aa733e75d930d34f0d318a6
fi
python src/prepare_simplerl_zoo_eval.py --src data/raw/simpleRL-reason/examples/simplelr_math_eval/data \
    --dst data/simplerl_zoo/eval
