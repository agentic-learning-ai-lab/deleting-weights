#!/usr/bin/env bash
# Fetch the upstream SDFT code and data (https://github.com/idanshen/Self-Distillation) at a pinned commit and place
# them next to the Deletion Tuning files in this directory. Run once from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=d77573212fa0a3ae2eeb64b9b44db1c251f75e3e
if [ ! -d upstream ]; then
  git clone https://github.com/idanshen/Self-Distillation.git upstream
fi
git -C upstream checkout --quiet "$COMMIT"
for f in main.py distil_trainer.py distil_config.py eval_tooluse.py eval_science.py; do
  cp upstream/$f $f
done
ln -sfn upstream/data data
echo "SDFT upstream ready at $COMMIT"
