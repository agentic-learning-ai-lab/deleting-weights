"""Evaluate SFT masks or full fine-tuned models, one per task, on the test split of MMLU-Pro or BBEH.

Each task's folder (``<trained_dir>/<task>/``, as train_sft.py saves it) holds either a Deletion Tuning mask, applied to
the base model, or a full fine-tuned model. It is scored on the task's test split, which the training seed determines:
MMLU-Pro picks the option with the lowest mean answer loss, and BBEH generates greedily and applies the official BBEH
scorer. Prints the accuracy per task and the mean over tasks. scripts/evaluate_sft.sh runs this with the paper's
settings.
"""

import argparse
import json
import os
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import apply_mask
import bbeh_eval
import sft_data
import train_sft

BATCH_SIZE = {"mmlu_pro": 4, "bbeh": 2}


def answer_loss(logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """Return each sample's mean next-token cross-entropy over its label tokens.

    Args:
        logits: [B, seq_len, vocab_size].
        label_ids: [B, seq_len], -100 where ignored.

    Returns:
        [B] losses.
    """
    B, seq_len, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()  # [B, seq_len-1, V]
    shift_labels = label_ids[:, 1:].contiguous()  # [B, seq_len-1]
    loss = nn.functional.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none")
    valid = (shift_labels != -100).float()
    return (loss.view(B, seq_len - 1) * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def eval_mmlu_pro(model: nn.Module, data: list[dict], collate_fn: Callable, accelerator: Accelerator) -> float:
    """Score every option of every question by its answer loss, pick the lowest, and return the accuracy (%)."""
    best: dict[int, tuple] = {}  # test_idx -> (loss, option, correct option)
    for start in range(0, len(data), BATCH_SIZE["mmlu_pro"]):
        batch = collate_fn(data[start: start + BATCH_SIZE["mmlu_pro"]])
        with accelerator.autocast():
            logits = model(input_ids=batch["input_ids"].to(accelerator.device),
                           attention_mask=batch["attention_mask"].to(accelerator.device)).logits
            losses = answer_loss(logits=logits, label_ids=batch["label_ids"].to(accelerator.device))
        for loss, test_idx, option, correct in zip(losses.tolist(), batch["test_idx"], batch["option"],
                                                   batch["correct_option"]):
            if test_idx not in best or loss < best[test_idx][0]:
                best[test_idx] = (loss, option, correct)
    correct = sum(p.strip().lower() == t.strip().lower() for _, p, t in best.values())
    return 100 * correct / len(best)


def eval_bbeh(model: nn.Module, data: list[dict], collate_fn: Callable, accelerator: Accelerator,
              tokenizer: PreTrainedTokenizerFast) -> float:
    """Generate greedy answers and return the accuracy (%) under the official BBEH scorer."""
    correct = 0
    for start in range(0, len(data), BATCH_SIZE["bbeh"]):
        batch = collate_fn(data[start: start + BATCH_SIZE["bbeh"]])
        input_ids = batch["input_ids"].to(accelerator.device)
        with accelerator.autocast():
            tokens = model.generate(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(accelerator.device),
                max_new_tokens=batch["generation_length"][0],
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(tokens[:, input_ids.shape[1]:], skip_special_tokens=True)
        correct += sum(bbeh_eval.fuzzy_match(prediction=bbeh_eval.preprocess_sample(text),
                                             reference=bbeh_eval.preprocess_reference(label))
                       for text, label in zip(texts, batch["label"]))
    return 100 * correct / len(data)


def load_model(path: str, accelerator: Accelerator) -> nn.Module:
    """Load a model in bf16 for evaluation."""
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    return accelerator.prepare(model).eval()


@torch.no_grad()
def main() -> None:
    """Evaluate each task's mask or model on the task's test split and print the scores."""
    parser = argparse.ArgumentParser(description="Evaluate SFT masks or full fine-tuned models, one per task.")
    parser.add_argument("--benchmark", type=str, required=True, choices=["mmlu_pro", "bbeh"])
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(train_sft.MODEL_NAME_TO_PATH))
    parser.add_argument("--seed", type=int, default=train_sft.DEFAULT_SEED,
                        help="Seed of the training run, which sets the test split")
    parser.add_argument("--trained_dir", type=str, required=True,
                        help="A run's masks/ or models/ directory, with one <task>/ folder per task")
    parser.add_argument("--data_dir", type=str, default=None, help="BBEH clone (BBEH only)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for eval_results.json")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer = train_sft.load_tokenizer(model_name=args.model_name)
    splits = train_sft.build_splits(benchmark=args.benchmark, tokenizer=tokenizer, seed=args.seed,
                                    data_dir=args.data_dir)
    if args.benchmark == "mmlu_pro":
        evaluate = partial(eval_mmlu_pro, collate_fn=partial(sft_data.collate_mmlu_pro_test,
                                                             pad_token_id=tokenizer.pad_token_id))
    else:
        evaluate = partial(eval_bbeh, tokenizer=tokenizer,
                           collate_fn=partial(sft_data.collate_bbeh_test, pad_token_id=tokenizer.pad_token_id))

    folders = [os.path.join(args.trained_dir, train_sft.task_folder(task)) for task in splits.tasks]
    full_models = not apply_mask.is_mask(path=folders[0])
    if not full_models:
        base = load_model(path=train_sft.MODEL_NAME_TO_PATH[args.model_name], accelerator=accelerator)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}

    scores = {}
    for task, folder in zip(splits.tasks, folders):
        if full_models:
            model = load_model(path=folder, accelerator=accelerator)
        else:
            base.load_state_dict(base_state)
            apply_mask.apply_mask(model=base, masks=apply_mask.load_mask(mask_dir=folder))
            model = base
        scores[task] = evaluate(model=model, data=[d for d in splits.test if d["task"] == task],
                                accelerator=accelerator)
        print(f"{task:<40} {scores[task]:.2f}")
    mean = sum(scores.values()) / len(scores)
    print(f"{'mean':<40} {mean:.2f}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump({"scores": scores, "mean": mean}, f, indent=2)


if __name__ == "__main__":
    main()
