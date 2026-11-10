"""Prepare the SimpleRL-Zoo training parquets (hkust-nlp/SimpleRL-Zoo-Data) for verl.

The ``simplelr_qwen_level3to5`` prompts carry the full Qwen chat template inside a single user message; they become
explicit system and user messages so the tokenizer's chat template renders the same qwen-boxed prompt once. The
``simplelr_abel_level1to4`` prompts (for Qwen2.5-0.5B) are kept as they are. Every row gets data_source
``simplerl_zoo_math`` so the SimpleRL-Zoo reward scores it.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

QWEN_RE = re.compile(
    r"<\|im_start\|>system\n"
    r"You are a helpful assistant\.<\|im_end\|>\n"
    r"<\|im_start\|>user\n"
    r"(?P<body>.*?)"
    r"<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n?$",
    flags=re.DOTALL,
)


def qwen_messages(prompt: list[dict]) -> list[dict]:
    """Return explicit system and user messages for a qwen-boxed prompt held in a single user message."""
    assert len(prompt) == 1 and prompt[0]["role"] == "user", f"unexpected prompt: {prompt}"
    match = QWEN_RE.match(prompt[0]["content"])
    assert match, f"qwen template did not match:\n{prompt[0]['content'][:400]!r}"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": match.group("body")},
    ]


def main() -> None:
    """Convert the training split of the abel and qwen prompt subsets."""
    parser = argparse.ArgumentParser(description="Prepare the SimpleRL-Zoo training parquets for verl.")
    parser.add_argument("--src", type=Path, required=True, help="Local copy of the hkust-nlp/SimpleRL-Zoo-Data dataset")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    for subset in ("simplelr_abel_level1to4", "simplelr_qwen_level3to5"):
        df = pd.read_parquet(args.src / subset / "train.parquet")
        prompts = [list(p) for p in df["prompt"]]
        df["prompt"] = [qwen_messages(prompt=p) for p in prompts] if "qwen" in subset else prompts
        df["data_source"] = "simplerl_zoo_math"
        dst = args.dst / subset / "train.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        print(f"wrote {dst} ({len(df)} rows)")


if __name__ == "__main__":
    main()
