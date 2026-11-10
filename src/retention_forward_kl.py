"""Forward KL from the base model to a GRPO Deletion Tuning mask on the training task.

Following Shenfeld et al. (2025, arXiv:2509.04259), forward KL is estimated on the evaluation prompts with one greedy
response y from the base model pi_0 per prompt:

    KL(pi_0(.|x) || pi(.|x)) ~ sum_t [log pi_0(y_t | x, y_<t) - log pi(y_t | x, y_<t)]

The reported value is the per-token mean (each prompt's sum divided by its response length, averaged over prompts).
The base model generates with vLLM; both models then score the responses with teacher forcing in bf16, with fp32
log-softmax.
"""

import argparse
import gc
import json
import os

import torch
import transformers

import apply_mask
import eval_grpo_common

SCORE_BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Forward KL from the base model to a masked model.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model (Hugging Face ID or path)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Mask directory")
    parser.add_argument("--val_parquets", type=str, nargs="+", required=True, help="Evaluation parquets of the task")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for forward_kl.json")
    return parser.parse_args()


@torch.inference_mode()
def response_logprobs(
    model: torch.nn.Module,
    pad_id: int,
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
) -> list[torch.Tensor]:
    """Return each response's token log-probabilities log p(y_t | x, y_<t) under the model, on the CPU."""
    out = []
    for start in range(0, len(prompt_ids), SCORE_BATCH_SIZE):
        seqs = [p + r for p, r in zip(prompt_ids[start:start + SCORE_BATCH_SIZE],
                                      response_ids[start:start + SCORE_BATCH_SIZE])]
        input_ids = torch.full((len(seqs), max(map(len, seqs))), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for k, s in enumerate(seqs):
            input_ids[k, :len(s)] = torch.tensor(s)
            attention_mask[k, :len(s)] = 1
        logits = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), use_cache=False).logits
        for k, i in enumerate(range(start, start + len(seqs))):
            p, r = len(prompt_ids[i]), len(response_ids[i])
            # Logits at positions [p - 1, p - 1 + r) predict the response tokens at [p, p + r).
            log_probs = torch.log_softmax(logits[k, p - 1:p - 1 + r].float(), dim=-1)
            target = torch.tensor(response_ids[i], device=log_probs.device)
            out.append(log_probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu())
    return out


def main() -> None:
    """Roll out the base model, score the responses under both models, and write the forward KL."""
    args = parse_args()
    samples = eval_grpo_common.build_val_samples(val_parquets=args.val_parquets)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    prompt_ids = eval_grpo_common.build_prompt_token_ids(samples=samples, tokenizer=tokenizer, skip_chat_template=False)
    outputs = eval_grpo_common.greedy_generate(model_path=args.model_name, prompt_ids=prompt_ids)
    response_ids = [list(out.outputs[0].token_ids) for out in outputs]
    gc.collect()  # releases vLLM's GPU memory
    torch.cuda.empty_cache()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda().eval()
    base = response_logprobs(model=model, pad_id=pad_id, prompt_ids=prompt_ids, response_ids=response_ids)
    apply_mask.apply_mask(model=model, masks=apply_mask.load_mask(mask_dir=args.mask_dir))
    masked = response_logprobs(model=model, pad_id=pad_id, prompt_ids=prompt_ids, response_ids=response_ids)

    per_prompt = [(b - m).mean().item() for b, m, r in zip(base, masked, response_ids) if r]
    forward_kl = sum(per_prompt) / len(per_prompt)
    print(f"forward KL per token: {forward_kl:.4f} ({len(per_prompt)} of {len(response_ids)} prompts with a response)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "forward_kl.json"), "w") as f:
        json.dump({"model_name": args.model_name, "mask_dir": args.mask_dir, "n_prompts": len(per_prompt),
                   "forward_kl_per_token_mean": forward_kl}, f, indent=2)


if __name__ == "__main__":
    main()
