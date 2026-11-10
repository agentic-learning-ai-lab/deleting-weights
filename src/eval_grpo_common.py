"""Shared helpers for evaluating GRPO Deletion Tuning masks.

Used by evaluate_grpo.py (task accuracy) and retention_forward_kl.py (forward KL from the base model). Prompts are
tokenized as in training and decoded greedily with vLLM.
"""

import os
from typing import Any

# Set before vLLM starts so its workers use spawn.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
import transformers

# Llama 3.2 chat templates insert a date; gsm8k.yaml fixes it to this one in training.
CHAT_TEMPLATE_DATE = "14 Apr 2026"
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 4608


def build_val_samples(val_parquets: list[str]) -> list[dict[str, Any]]:
    """Read verl-format validation parquets into a list of prompts, data sources, and ground truths."""
    samples: list[dict[str, Any]] = []
    for vp in val_parquets:
        df = pd.read_parquet(vp)
        for _, row in df.iterrows():
            rm = row["reward_model"]
            if hasattr(rm, "item"):
                rm = rm.item()
            samples.append({
                "prompt": [dict(msg) for msg in list(row["prompt"])],
                "data_source": str(row["data_source"]),
                "ground_truth": str(rm["ground_truth"]),
            })
    return samples


def build_prompt_token_ids(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    skip_chat_template: bool,
) -> list[list[int]]:
    """Tokenize every prompt as in training: the raw user message with ``skip_chat_template``, else chat-templated."""
    token_ids = []
    for sample in samples:
        msgs = sample["prompt"]
        if skip_chat_template:
            assert len(msgs) == 1 and msgs[0]["role"] == "user", f"expected a single user message, got {msgs!r}"
            token_ids.append(tokenizer(msgs[0]["content"], add_special_tokens=False)["input_ids"])
        else:
            token_ids.append(tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, date_string=CHAT_TEMPLATE_DATE,
            ))
    return token_ids


def greedy_generate(model_path: str, prompt_ids: list[list[int]]) -> list[Any]:
    """Decode every prompt greedily in vLLM (bf16, at most MAX_NEW_TOKENS new tokens) and return its RequestOutputs."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.7, max_model_len=MAX_MODEL_LEN)
    return llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
    )
