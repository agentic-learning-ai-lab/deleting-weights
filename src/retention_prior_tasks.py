"""Prior-task accuracy of a GRPO Deletion Tuning mask, which measures forgetting.

The mask is applied to the base model and lm-eval-harness scores six benchmarks unrelated to the training task,
following Shenfeld et al. (2025, arXiv:2509.04259): HellaSwag, TruthfulQA (MC2), MMLU, IFEval, Winogrande, and
HumanEval, zero-shot in bf16 with the chat template. The prior-task score is the mean of each benchmark's primary
metric. With --checkpoint base, the base model is evaluated.
"""

import argparse
import json
import os

# Opt-in required by the `code_eval` metric of HumanEval.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

from lm_eval import evaluator

import apply_mask

# lm-eval-harness task of each benchmark and its primary metric.
TASK_TO_METRIC = {
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "mmlu": "acc,none",
    "ifeval": "inst_level_loose_acc,none",
    "winogrande": "acc,none",
    "humaneval_instruct": "pass@1,create_test",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prior-task accuracy on six lm-eval-harness benchmarks.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model (Hugging Face ID or path)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Mask directory or base")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for prior_tasks.json")
    return parser.parse_args()


def main() -> None:
    """Apply the mask, run lm-eval-harness, and write the score per benchmark and their mean."""
    args = parse_args()
    with apply_mask.model_path(base_model=args.model_name, checkpoint=args.checkpoint) as path:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={path},dtype=bfloat16",
            tasks=list(TASK_TO_METRIC),
            batch_size="auto",
            device="cuda",
            apply_chat_template=True,
            confirm_run_unsafe_code=True,
        )

    scores = {task: results["results"][task][metric] for task, metric in TASK_TO_METRIC.items()}
    mean = sum(scores.values()) / len(scores)
    for task, score in scores.items():
        print(f"{task:<20} {100 * score:.2f}")
    print(f"{'mean':<20} {100 * mean:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prior_tasks.json"), "w") as f:
        json.dump({"model_name": args.model_name, "checkpoint": args.checkpoint, "scores": scores,
                   "prior_task_mean": mean}, f, indent=2)


if __name__ == "__main__":
    main()
