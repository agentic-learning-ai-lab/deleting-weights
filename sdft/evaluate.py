"""Evaluate an SDFT checkpoint on ToolAlpaca (tooluse) or Science.

The checkpoint is a Deletion Tuning mask (applied to the base model), a full fine-tuning checkpoint, or "base". The SDFT
authors' evaluation script for the task (eval_tooluse.py or eval_science.py) scores it and writes eval_results.json.
scripts/evaluate_sdft.sh runs this with the paper's base model:
  python sdft/evaluate.py --dataset_name tooluse --model_name Qwen/Qwen2.5-7B-Instruct \\
      --checkpoint <run>/mask_step_200 --output_dir <out>
"""

import argparse
import json
import os
import subprocess
import sys

import apply_mask

SDFT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Evaluate an SDFT checkpoint")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["tooluse", "science"])
    parser.add_argument("--model_name", type=str, required=True, help="Base model (Hugging Face ID or path)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Mask directory, full fine-tuning checkpoint directory, or base")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for eval_results.json")
    return parser.parse_args()


def main() -> None:
    """Run the task evaluation on the checkpoint and print its accuracy."""
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint == "base" else os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    with apply_mask.model_path(base_model=args.model_name, checkpoint=checkpoint) as path:
        # The evaluation scripts read their data relative to the SDFT directory.
        command = [sys.executable, f"eval_{args.dataset_name}.py", "--model_path", path, "--output_dir", output_dir]
        subprocess.run(command, check=True, cwd=SDFT_DIR)
    with open(os.path.join(output_dir, "eval_results.json")) as f:
        print(f"{args.dataset_name} accuracy: {100 * json.load(f)['accuracy']:.2f}")


if __name__ == "__main__":
    main()
