"""Full fine-tuning with on-policy self-distillation (the SDFT baseline), with the authors' settings.

Trains with the configuration of the authors' main.py (https://github.com/idanshen/Self-Distillation, copied here by
setup.sh), reports to no experiment tracker, and saves the model weights without optimizer state every --save_steps
optimizer steps and after the last step to <output_dir>/checkpoint-<N>/, which scripts/evaluate_sdft.sh evaluates.

Usage: python train_fullparam.py --dataset_name <tooluse|science> --model_name Qwen/Qwen2.5-7B-Instruct
                                 --output_dir <dir> --learning_rate 5e-5 --num_train_epochs 2 --save_steps <N>
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer
import main as sdft_main


def parse_args() -> argparse.Namespace:
    """Parse the command line (main.py's flags plus the checkpoint cadence)."""
    parser = argparse.ArgumentParser(description="Full fine-tuning with on-policy self-distillation")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["tooluse", "science"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--save_steps", type=int, required=True, help="optimizer steps between saved checkpoints")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    """Train the model with SDFT and save its checkpoints."""
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    load_dataset = {"tooluse": sdft_main.load_tooluse_dataset, "science": sdft_main.load_science_dataset}
    dataset, _ = load_dataset[args.dataset_name](args.seed)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.3,
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
        save_steps=args.save_steps,
        save_only_model=True,
        max_grad_norm=1,
        report_to="none",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )
    trainer = DistilTrainer(
        model=model, ref_model=teacher_model, args=config, train_dataset=dataset, processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
