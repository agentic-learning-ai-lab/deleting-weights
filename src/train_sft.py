"""Train Deletion Tuning masks with supervised fine-tuning on MMLU-Pro or BBEH.

For each task (the 14 MMLU-Pro categories or the 23 BBEH tasks), the script:

1. splits the examples into a 25% test set and a train set after a seeded shuffle;
2. initializes one mask logit per gate-projection weight from Wanda scores on the train set, so that about
   `--init_sparsity` of the weights start deleted;
3. trains only the logits on the answer-token cross-entropy, while the mask sharpness rises linearly from
   `--sharpness_start` to `--sharpness_end` and the weight of the sparse-weight regularizer decays linearly from
   `--weight_reg` to 0;
4. saves the mask to `<output_dir>/masks/<task>/`.

With `--fullparam`, every weight trains instead (the full fine-tuning baseline) and each task's model is saved to
`<output_dir>/models/<task>/`. src/evaluate_sft.py evaluates either folder.

Example (one GPU; scripts/train_sft.sh has the paper's settings):
  accelerate launch --num_processes 1 --mixed_precision bf16 src/train_sft.py \\
      --benchmark mmlu_pro --model_name llama1b --output_dir outputs/sft/mmlu_pro_llama1b ...
"""

import argparse
import gc
import logging
import os
import pprint
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import sft_data
import wanda_calibration
from learnable_sparsity.differentiable_mask import DifferentiableMask
from learnable_sparsity.sparse_linear import (
    SparseLinear,
    convert_to_sparse_model,
    extract_all_masks,
    save_mask,
    set_weight_norm,
    step_all_sparse_modules,
)

logger = get_logger(__name__, log_level="INFO")

MODEL_NAME_TO_PATH: dict[str, str] = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}
MAX_SEQ_LEN: dict[str, int] = {"mmlu_pro": 2048, "bbeh": 8192}
EVAL_RATIO = 0.25
DEFAULT_SEED = 0
TARGET_MODULES = ["gate_proj"]
LOG_EVERY = 10

Splits = sft_data.MMLUProSplits | sft_data.BBEHSplits


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load the model's tokenizer with left padding, padding with the EOS token."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[model_name])
    tokenizer.padding_side = "left"
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.pad_token is None
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_splits(benchmark: str, tokenizer: PreTrainedTokenizerFast, seed: int, data_dir: str | None) -> Splits:
    """Split every task of the benchmark into train and test with the seed."""
    if benchmark == "mmlu_pro":
        return sft_data.MMLUProSplits(seed=seed, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN[benchmark],
                                      eval_ratio=EVAL_RATIO)
    assert data_dir is not None, "--data_dir is required for bbeh"
    return sft_data.BBEHSplits(data_dir=data_dir, seed=seed, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN[benchmark],
                               eval_ratio=EVAL_RATIO)


def task_folder(task: str) -> str:
    """Return the folder name of a task's mask or model, with spaces in the task name replaced by underscores."""
    return task.replace(" ", "_")


def sum_sparse_weight_norms(model: nn.Module) -> torch.Tensor:
    """Sum the regularizer terms ||W * m||^2 that the SparseLinear layers stored in their last forward."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, SparseLinear) and hasattr(module, "sparse_weight_norm"):
            total = total + module.sparse_weight_norm.float()
    return total


@torch.enable_grad()
def train_task(args: argparse.Namespace, task: str, splits: Splits, model: nn.Module, accelerator: Accelerator) -> None:
    """Train one task's mask, or every weight with --fullparam."""
    train_set = splits.train_set(task=task)
    assert len(train_set) > 0, f"no training samples for {task}"
    collate_fn = partial(sft_data.collate_train, pad_token_id=splits.tokenizer.pad_token_id)
    batch_size = min(args.train_batch_size, len(train_set))
    iters = args.train_steps * args.grad_accum_steps  # micro-batches

    if args.fullparam:
        for p in model.parameters():
            p.requires_grad = True
    elif args.prior_strength > 0:
        calib_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        wanda_scores = wanda_calibration.compute_wanda_scores(
            model=model, dataloader=calib_loader, device=accelerator.device, num_samples=len(train_set),
            target_modules=TARGET_MODULES,
        )
        wanda_calibration.init_masks_from_wanda(
            model=model, wanda_scores=wanda_scores, init_sparsity=args.init_sparsity,
            prior_strength=args.prior_strength,
        )
    weight_reg = 0.0 if args.fullparam else args.weight_reg
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optim.zero_grad()
    logger.info(f"[{task}] training: {len(train_set)} samples, batch_size={batch_size}, {iters} iters, "
                f"{sum(p.numel() for p in trainable_params):,d} trainable params")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
                              num_workers=0 if len(train_set) < 256 else 4)

    model.train()
    curr_iter = 0
    while curr_iter < iters:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            label_ids = batch["label_ids"].to(accelerator.device)
            with accelerator.autocast():
                # Left padding: positions count from the first real token.
                seq_len = input_ids.shape[1]
                seq_range = torch.arange(seq_len, device=input_ids.device, dtype=torch.int64)  # [seq_len]
                offsets = (seq_len - attention_mask.sum(dim=1)).unsqueeze(1)  # [B, 1]
                position_ids = (seq_range.unsqueeze(0) - offsets).clamp(min=0)  # [B, seq_len]
                ce_loss = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, position_ids=position_ids,
                ).loss
                loss = ce_loss
                if weight_reg > 0:
                    loss = loss - weight_reg * (1.0 - curr_iter / iters) * sum_sparse_weight_norms(model=model)
            accelerator.backward(loss / args.grad_accum_steps)

            if (curr_iter + 1) % args.grad_accum_steps == 0:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()
                optim.zero_grad()
                step_all_sparse_modules(model=model)
            if (curr_iter + 1) % LOG_EVERY == 0:
                logger.info(f"[{task}] iter {curr_iter + 1}/{iters} | loss={ce_loss.item():.4f}")
            curr_iter += 1
            if curr_iter >= iters:
                break


@torch.no_grad()
def train_all_tasks(args: argparse.Namespace, model: nn.Module, splits: Splits, accelerator: Accelerator) -> None:
    """Train one mask per task from the same initial logits (with --fullparam, one model per task) and save each."""
    if args.fullparam:
        # fp32 master weights; the forward pass runs under bf16 autocast.
        model.float()
    else:
        convert_to_sparse_model(
            model=model,
            target_modules=TARGET_MODULES,
            sharpness=[args.sharpness_start, args.sharpness_end],
            total_steps=args.train_steps,
        )
        set_weight_norm(model=model, enabled=args.weight_reg > 0)
    initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    for task_idx, task in enumerate(splits.tasks):
        logger.info(f"TASK {task_idx + 1}/{len(splits.tasks)}: {task}")
        state = model.state_dict()
        for k in state:
            state[k].copy_(initial_state[k])
        torch.cuda.empty_cache()
        for module in model.modules():
            if isinstance(module, DifferentiableMask):
                module.current_step = 0
        model.eval()
        train_task(args=args, task=task, splits=splits, model=model, accelerator=accelerator)
        gc.collect()

        if args.fullparam:
            model_dir = os.path.join(args.output_dir, "models", task_folder(task))
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(model_dir, state_dict={k: v.to(torch.bfloat16)
                                                             for k, v in unwrapped.state_dict().items()})
            logger.info(f"[{task}] model saved to {model_dir}")
        else:
            mask_dir = os.path.join(args.output_dir, "masks", task_folder(task))
            save_mask(mask=extract_all_masks(model=model), save_path=mask_dir)
            logger.info(f"[{task}] mask saved to {mask_dir}")


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Deletion Tuning with supervised fine-tuning on MMLU-Pro or BBEH")
    parser.add_argument("--benchmark", type=str, required=True, choices=["mmlu_pro", "bbeh"])
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(MODEL_NAME_TO_PATH))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="BBEH only: root of a clone of https://github.com/google-deepmind/bbeh")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sets each task's train/test split")
    parser.add_argument("--train_steps", type=int, required=True, help="Optimizer steps per task")
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, required=True, help="Constant learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fullparam", action="store_true",
                        help="Full fine-tuning baseline: train every weight (the mask flags below are then unused)")
    parser.add_argument("--init_sparsity", type=float, required=True,
                        help="Fraction of weights deleted at initialization")
    parser.add_argument("--prior_strength", type=float, default=3.0, help="Strength of the Wanda prior on the logits")
    parser.add_argument("--sharpness_start", type=float, required=True, help="Initial mask sharpness")
    parser.add_argument("--sharpness_end", type=float, required=True, help="Final mask sharpness")
    parser.add_argument("--weight_reg", type=float, required=True,
                        help="Initial weight of the sparse-weight regularizer (decays linearly to 0)")
    return parser.parse_args()


def main() -> None:
    """Train and save one mask per task, or with --fullparam one model per task."""
    args = parse_args()
    accelerator = Accelerator()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info(f"Arguments:\n{pprint.pformat(vars(args), indent=2)}")

    tokenizer = load_tokenizer(model_name=args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_TO_PATH[args.model_name], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = accelerator.prepare(model)
    for p in model.parameters():
        p.requires_grad = False
    splits = build_splits(benchmark=args.benchmark, tokenizer=tokenizer, seed=args.seed, data_dir=args.data_dir)
    logger.info(f"{args.benchmark}: {len(splits.test)} test samples, max_seq_len={splits.max_seq_len}")
    train_all_tasks(args=args, model=model, splits=splits, accelerator=accelerator)


if __name__ == "__main__":
    main()
