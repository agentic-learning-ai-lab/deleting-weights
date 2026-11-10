"""Build the six SimpleRL-Zoo validation parquets from the simpleRL-reason evaluation data.

Each benchmark (GSM8K, MATH500, Minerva Math, OlympiadBench, AIME24, AMC23) gets two prompt
formats: ``qwen`` (system and user messages ending with the boxed-answer instruction, rendered
by the chat template) and ``abel`` (one plain "Question: ... Answer: Let's think step by step."
message). Questions and ground truths are parsed with SimpleRL-Zoo's own parser, and
data_source is ``simplerl_zoo_<benchmark>`` so the SimpleRL-Zoo reward scores them.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from verl.utils.reward_score.qwen_math_eval_toolkit.parser import parse_ground_truth, parse_question

# Benchmark name -> simpleRL-reason dataset (its data directory and answer parser).
BENCHMARKS = {
    "gsm8k": "gsm8k",
    "math500": "math500",
    "minerva": "minerva_math",
    "olympiadbench": "olympiadbench",
    "aime24": "aime24",
    "amc23": "amc23",
}
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def qwen_prompt(question: str) -> list[dict]:
    """Return the qwen-boxed chat messages for a question."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{question}\n{BOXED_INSTRUCTION}"},
    ]


def abel_prompt(question: str) -> list[dict]:
    """Return the single plain abel-format message for a question."""
    return [{"role": "user", "content": f"Question:\n{question}\nAnswer:\nLet's think step by step.\n"}]


PROMPTS = {"qwen": qwen_prompt, "abel": abel_prompt}


def main() -> None:
    """Write every benchmark in both prompt formats."""
    parser = argparse.ArgumentParser(description="Build the SimpleRL-Zoo validation parquets.")
    parser.add_argument("--src", type=Path, required=True,
                        help="simpleRL-reason examples/simplelr_math_eval/data directory")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output directory (one subdirectory per prompt format)")
    args = parser.parse_args()

    for template, make_prompt in PROMPTS.items():
        for bench, dataset in BENCHMARKS.items():
            with open(args.src / dataset / "test.jsonl") as f:
                examples = [json.loads(line) for line in f if line.strip()]
            rows = []
            for ex in examples:
                question = parse_question(ex, dataset)
                if question:
                    rows.append({
                        "data_source": f"simplerl_zoo_{bench}",
                        "prompt": make_prompt(question=question),
                        "ability": "math",
                        "reward_model": {"ground_truth": str(parse_ground_truth(ex, dataset)[1]), "style": "rule"},
                    })
            out = args.dst / template / f"{bench}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
