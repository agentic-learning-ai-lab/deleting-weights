"""Train a Deletion Tuning mask with on-policy self-distillation (SDFT).

Every gate_proj weight of the student gets one learnable logit, and the base weights stay frozen. The logits start
from Wanda scores and are trained with the self-distillation loss of the SDFT authors' DistilTrainer, while vLLM
samples from the hard-masked student. The binary masks are written to <output_dir>/mask_step_<N>/ every --save_steps
optimizer steps and to <output_dir>/mask/ at the end.

Run from this directory after setup.sh, with the repository's src/ on PYTHONPATH (scripts/train_sdft.sh does both):
  python train.py --dataset_name tooluse --model_name Qwen/Qwen2.5-7B-Instruct --output_dir <out> \\
      --learning_rate 5e-3 --sharpness_end 10000 --save_steps 100
"""

import argparse
import os

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import wanda_calibration
from distil_config import DistilConfig
from learnable_sparsity import sparse_linear
from main import load_science_dataset, load_tooluse_dataset
from mask_aware_distil_trainer import MaskAwareDistilTrainer

TARGET_MODULES = ["gate_proj"]
CALIB_BATCH_SIZE = 8  # prompts per Wanda calibration batch


class MaskSaveCallback(TrainerCallback):
    """Save the student mask to <output_dir>/mask_step_<N>/ every save_steps optimizer steps."""

    def __init__(self, student: torch.nn.Module, save_steps: int, output_dir: str) -> None:
        """Store the student, the save interval, and the output directory."""
        self.student = student
        self.save_steps = save_steps
        self.output_dir = output_dir

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    **kwargs) -> None:
        """Save the mask when the global step is a multiple of save_steps."""
        if state.global_step % self.save_steps == 0:
            mask_dir = os.path.join(self.output_dir, f"mask_step_{state.global_step}")
            masks = sparse_linear.extract_all_masks(model=self.student)
            self.student.train()  # extract_all_masks leaves the student in eval mode
            sparse_linear.save_mask(mask=masks, save_path=mask_dir)
            print(f"[train] saved {len(masks)} masks at step {state.global_step} to {mask_dir}")


class MaskScheduleCallback(TrainerCallback):
    """Advance the mask sharpness schedule of the student and the teacher once per optimizer step."""

    def __init__(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        """Store the student and teacher whose schedules advance together."""
        self.student = student
        self.teacher = teacher

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    **kwargs) -> None:
        """Step every mask schedule."""
        sparse_linear.step_all_sparse_modules(model=self.student)
        sparse_linear.step_all_sparse_modules(model=self.teacher)


def parse_args() -> argparse.Namespace:
    """Parse the command line; defaults are the settings shared by the paper's SDFT runs."""
    parser = argparse.ArgumentParser(description="Deletion Tuning with on-policy self-distillation")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["tooluse", "science"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--sharpness_end", type=float, required=True, help="final mask sharpness")
    parser.add_argument("--save_steps", type=int, required=True, help="optimizer steps between saved masks")
    parser.add_argument("--sharpness_start", type=float, default=100.0, help="initial mask sharpness")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="prompts per optimizer step")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="teacher EMA rate toward the student")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.15)
    parser.add_argument("--init_sparsity", type=float, default=1.0 / 2048, help="fraction of weights deleted at init")
    parser.add_argument("--prior_strength", type=float, default=3.0, help="Wanda prior shift in units of the logit std")
    parser.add_argument("--n_calib_examples", type=int, default=32, help="training prompts used for Wanda calibration")
    return parser.parse_args()


def wanda_scores_on_prompts(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, prompts: list[list[dict]],
                            device: torch.device) -> dict[str, torch.Tensor]:
    """Return the dense model's Wanda scores on chat-templated prompts."""
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    batches = []
    for start in range(0, len(texts), CALIB_BATCH_SIZE):
        enc = tokenizer(texts[start:start + CALIB_BATCH_SIZE], return_tensors="pt", padding=True, truncation=True,
                        max_length=1024)
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    print(f"[train] Wanda calibration on {len(texts)} prompts (batch size {CALIB_BATCH_SIZE})")
    return wanda_calibration.compute_wanda_scores(model=model, dataloader=batches, device=device,
                                                  num_samples=len(texts), target_modules=TARGET_MODULES)


def share_frozen_weights(student: torch.nn.Module, teacher: torch.nn.Module) -> None:
    """Point every teacher parameter except its mask logits at the student's tensor and free the teacher's copy."""
    student_modules = dict(student.named_modules())
    for name, module in teacher.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if pname == "gate":  # the teacher's own mask logits
                continue
            source = getattr(student_modules[name], pname)
            assert source.shape == param.shape and torch.equal(source, param), (
                f"{name}.{pname} differs from the student's"
            )
            setattr(module, pname, source)
    torch.cuda.empty_cache()


def main() -> None:
    """Train the mask and write the final compressed mask to <output_dir>/mask/."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    print(f"[train] model = {args.model_name}; dataset = {args.dataset_name}; lr = {args.learning_rate}, "
          f"epochs = {args.num_train_epochs}, prompts per step = {args.num_prompts_per_batch}, "
          f"teacher EMA = {args.ref_model_mixup_alpha}")
    print(f"[train] target = {TARGET_MODULES}, init_sparsity = {args.init_sparsity:.6f}, "
          f"prior_strength = {args.prior_strength}, sharpness = [{args.sharpness_start}, {args.sharpness_end}]")

    student = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    load_dataset = {"tooluse": load_tooluse_dataset, "science": load_science_dataset}[args.dataset_name]
    dataset, _ = load_dataset(args.seed)
    total_steps = args.num_train_epochs * (len(dataset) // args.num_prompts_per_batch)
    print(f"[train] dataset size = {len(dataset)}; total steps = {total_steps}")

    for net in (student, teacher):
        sparse_linear.convert_to_sparse_model(
            model=net,
            target_modules=TARGET_MODULES,
            sharpness=[args.sharpness_start, args.sharpness_end],
            total_steps=total_steps,
        )
        sparse_linear.set_weight_norm(model=net, enabled=False)

    wanda_scores = wanda_scores_on_prompts(
        model=student, tokenizer=tokenizer, device=device,
        prompts=[dataset[i]["prompt"] for i in range(min(args.n_calib_examples, len(dataset)))],
    )
    wanda_calibration.init_masks_from_wanda(
        model=student, wanda_scores=wanda_scores, init_sparsity=args.init_sparsity, prior_strength=args.prior_strength,
    )
    # The teacher starts from the student's logits.
    teacher_state = teacher.state_dict()
    for k, v in student.state_dict().items():
        if "diff_mask.gate" in k:
            teacher_state[k].copy_(v)
    del teacher_state
    share_frozen_weights(student=student, teacher=teacher)

    for net in (student, teacher):
        for name, param in net.named_parameters():
            if "diff_mask.gate" not in name:
                param.requires_grad_(False)
        # With frozen embeddings and gradient checkpointing, the first block's input must require grad.
        net.enable_input_require_grads()

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_strategy="no",
        save_total_limit=0,
        max_grad_norm=1,
        report_to="none",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = MaskAwareDistilTrainer(
        model=student, ref_model=teacher,
        args=config, train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            MaskSaveCallback(student=student, save_steps=args.save_steps, output_dir=args.output_dir),
            MaskScheduleCallback(student=student, teacher=teacher),
        ],
    )
    trainer.train()

    mask_dir = os.path.join(args.output_dir, "mask")
    sparse_linear.save_mask(mask=sparse_linear.extract_all_masks(model=student), save_path=mask_dir)
    print(f"[train] saved final mask to {mask_dir}")


if __name__ == "__main__":
    main()
