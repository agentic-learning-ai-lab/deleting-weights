"""Evaluate a GRPO checkpoint on GSM8K or the six SimpleRL-Zoo benchmarks.

The checkpoint is a Deletion Tuning mask (applied to the base model), a full fine-tuned model, or "base". vLLM decodes
greedily (at most 4096 new tokens) from prompts tokenized as in training, and each response is scored with the training
reward of its data source. scripts/evaluate_grpo.sh runs this with each model's settings.
"""

import argparse
import json
import os
from collections import defaultdict

import transformers

import apply_mask
import eval_grpo_common
from verl.utils.reward_score import default_compute_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a GRPO checkpoint.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model (Hugging Face ID or path)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Mask directory, model directory, or base")
    parser.add_argument("--val_parquets", type=str, nargs="+", required=True, help="Validation parquets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for eval_results.json")
    parser.add_argument("--skip_chat_template", action="store_true",
                        help="Tokenize the user message as raw text (for runs trained with data.skip_chat_template)")
    return parser.parse_args()


def main() -> None:
    """Generate, score, and write the accuracy per benchmark."""
    args = parse_args()
    samples = eval_grpo_common.build_val_samples(val_parquets=args.val_parquets)
    with apply_mask.model_path(base_model=args.model_name, checkpoint=args.checkpoint) as path:
        prompt_ids = eval_grpo_common.build_prompt_token_ids(
            samples=samples, tokenizer=transformers.AutoTokenizer.from_pretrained(path),
            skip_chat_template=args.skip_chat_template,
        )
        outputs = eval_grpo_common.greedy_generate(model_path=path, prompt_ids=prompt_ids)

    task_accs: dict[str, list[float]] = defaultdict(list)
    for sample, output in zip(samples, outputs):
        score = default_compute_score(
            data_source=sample["data_source"], solution_str=output.outputs[0].text, ground_truth=sample["ground_truth"],
        )
        task_accs[sample["data_source"]].append(score)

    per_task = {task: sum(accs) / len(accs) for task, accs in sorted(task_accs.items())}
    mean = sum(per_task.values()) / len(per_task)
    for task, acc in per_task.items():
        print(f"{task:<30} {100 * acc:.2f}")
    print(f"{'mean':<30} {100 * mean:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump({"model_name": args.model_name, "checkpoint": args.checkpoint, "accuracy": per_task, "mean": mean},
                  f, indent=2)


if __name__ == "__main__":
    main()
