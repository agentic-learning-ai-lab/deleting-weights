"""Train/test splits and batching for supervised fine-tuning on MMLU-Pro and BBEH.

Each benchmark is split per task into a test set (the first `eval_ratio` fraction after a seeded shuffle) and a
train set (the rest). Training samples are a question followed by its answer, with the loss on the answer tokens only.
Batches are left-padded.

- MMLU-Pro (TIGER-Lab/MMLU-Pro, test split, 14 categories): `{question}\\n{answer}`; evaluation scores every option
  by its mean answer-token loss and picks the lowest.
- BBEH (https://github.com/google-deepmind/bbeh, 23 tasks): `{input}\\nAnswer: {target}`; evaluation generates
  greedily and matches the answer with the official BBEH scorer (`bbeh_eval.py`).
"""

import json
import math
import os
import random

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from bbeh_tasks import BBEH_TASKS

MMLU_PRO_DELIMITER = "\n"
BBEH_ANSWER_PREFIX = "\nAnswer: "
MAX_PART_LEN = 1024  # truncation of each MMLU-Pro question and answer, in tokens


def pad_left(sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    """Left-pad 1-D tensors into a [B, max_len] batch."""
    padded = pad_sequence([seq.flip(0) for seq in sequences], batch_first=True, padding_value=padding_value)
    return padded.flip(1)


def tokenize(text: str, tokenizer: PreTrainedTokenizerFast, max_length: int) -> torch.Tensor:
    """Tokenize text (truncated to max_length tokens) into 1-D ids without the leading BOS token."""
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    assert input_ids[0].item() == tokenizer.bos_token_id
    return input_ids[1:]


def encode(prompt_ids: torch.Tensor, answer_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return input_ids, attention_mask and label_ids for prompt + answer, with the loss on the answer only."""
    input_ids = torch.cat([prompt_ids, answer_ids])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "label_ids": torch.cat([torch.full_like(prompt_ids, -100), answer_ids]),
    }


def encode_mmlu_pro(question: str, answer: str, tokenizer: PreTrainedTokenizerFast,
                    delimiter_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encode `{question}{delimiter}{answer}`, truncating the question and the answer separately."""
    prompt_ids = torch.cat([tokenize(text=question, tokenizer=tokenizer, max_length=MAX_PART_LEN), delimiter_ids])
    return encode(prompt_ids=prompt_ids, answer_ids=tokenize(text=answer, tokenizer=tokenizer, max_length=MAX_PART_LEN))


def collate_train(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad training samples into input_ids, attention_mask and label_ids tensors of shape [B, max_len]."""
    return {
        "input_ids": pad_left([x["input_ids"] for x in batch], padding_value=pad_token_id),
        "attention_mask": pad_left([x["attention_mask"] for x in batch], padding_value=0),
        "label_ids": pad_left([x["label_ids"] for x in batch], padding_value=-100),
    }


def collate_mmlu_pro_test(batch: list[dict], pad_token_id: int) -> dict:
    """Pad MMLU-Pro option samples for loss scoring and keep their question index and option labels."""
    return {
        **collate_train(batch=batch, pad_token_id=pad_token_id),
        "test_idx": [x["test_idx"] for x in batch],
        "option": [x["option"] for x in batch],
        "correct_option": [x["correct_option"] for x in batch],
    }


def collate_bbeh_test(batch: list[dict], pad_token_id: int) -> dict:
    """Pad BBEH prompts for generation and keep their labels and generation budget."""
    return {
        "input_ids": pad_left([x["input_ids"] for x in batch], padding_value=pad_token_id),
        "attention_mask": pad_left([x["attention_mask"] for x in batch], padding_value=0),
        "label": [x["label"] for x in batch],
        "generation_length": [x["generation_length"] for x in batch],
    }


class MMLUProSplits:
    """MMLU-Pro split per category: `test` holds one tokenized sample per (test question, option) and
    `train_pairs` the raw training questions of each category."""

    def __init__(self, seed: int, tokenizer: PreTrainedTokenizerFast, max_seq_len: int, eval_ratio: float) -> None:
        """Load MMLU-Pro, split each category with a seeded shuffle and tokenize every test option."""
        self.tokenizer = tokenizer
        self.delimiter_ids = tokenize(text=MMLU_PRO_DELIMITER, tokenizer=tokenizer, max_length=MAX_PART_LEN)
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
        self.tasks = sorted(set(dataset["category"]))
        questions = [{"task": d["category"], "input": d["question"], "output": d["options"][d["answer_index"]],
                      "options": d["options"]} for d in dataset]

        self.train_pairs: dict[str, list[dict]] = {}
        self.test: list[dict] = []
        for task in self.tasks:
            task_questions = [q for q in questions if q["task"] == task]
            random.seed(seed)
            random.shuffle(task_questions)
            num_test = math.ceil(len(task_questions) * eval_ratio)
            self.train_pairs[task] = task_questions[num_test:]
            random.seed(seed)
            random.shuffle(self.train_pairs[task])
            for test_idx, question in enumerate(task_questions[:num_test]):
                options = [self._encode_option(question=question, option=o, test_idx=test_idx, max_seq_len=max_seq_len)
                           for o in question["options"]]
                if None not in options:
                    self.test += options
        # Training samples longer than the longest test sample are dropped.
        self.max_seq_len = max(len(d["input_ids"]) for d in self.test)

    def _encode_option(self, question: dict, option: str, test_idx: int, max_seq_len: int) -> dict | None:
        """Encode the question with one option as the answer, or return None if it is longer than max_seq_len."""
        sample = encode_mmlu_pro(question=question["input"], answer=option, tokenizer=self.tokenizer,
                                 delimiter_ids=self.delimiter_ids)
        if len(sample["input_ids"]) > max_seq_len:
            return None
        return {**sample, "task": question["task"], "test_idx": test_idx, "option": option,
                "correct_option": question["output"]}

    def train_set(self, task: str) -> list[dict]:
        """Encode one category's training questions with their correct answers."""
        samples = [encode_mmlu_pro(question=p["input"], answer=p["output"], tokenizer=self.tokenizer,
                                   delimiter_ids=self.delimiter_ids) for p in self.train_pairs[task]]
        return [s for s in samples if len(s["input_ids"]) <= self.max_seq_len]


class BBEHSplits:
    """BBEH split per task: `test` holds the tokenized `{input}\\nAnswer: ` prompts and `train_pairs` the raw
    training examples of each task."""

    def __init__(self, data_dir: str, seed: int, tokenizer: PreTrainedTokenizerFast, max_seq_len: int,
                 eval_ratio: float) -> None:
        """Read every `bbeh/benchmark_tasks/<task>/task.json` under data_dir, split it and tokenize test prompts."""
        self.tokenizer = tokenizer
        self.tasks = list(BBEH_TASKS)
        self.train_pairs: dict[str, list[dict]] = {}
        self.test: list[dict] = []
        for task in self.tasks:
            with open(os.path.join(data_dir, "bbeh", "benchmark_tasks", task, "task.json")) as f:
                examples = json.load(f)["examples"]
            random.seed(seed)
            random.shuffle(examples)
            num_test = math.ceil(len(examples) * eval_ratio)
            self.train_pairs[task] = examples[num_test:]
            for example in examples[:num_test]:
                prompt_ids = tokenize(text=example["input"] + BBEH_ANSWER_PREFIX, tokenizer=tokenizer,
                                      max_length=max_seq_len)
                if len(prompt_ids) + BBEH_TASKS[task] <= max_seq_len:
                    self.test.append({"task": task, "label": example["target"], "input_ids": prompt_ids,
                                      "attention_mask": torch.ones_like(prompt_ids),
                                      "generation_length": BBEH_TASKS[task]})
        # Training samples longer than the longest test prompt plus its generation budget are dropped.
        self.max_seq_len = max(len(d["input_ids"]) + d["generation_length"] for d in self.test)

    def train_set(self, task: str) -> list[dict]:
        """Encode one task's training examples as `{input}\\nAnswer: {target}` with the loss on the target only."""
        samples = [encode(prompt_ids=tokenize(text=p["input"] + BBEH_ANSWER_PREFIX, tokenizer=self.tokenizer,
                                              max_length=self.max_seq_len),
                          answer_ids=tokenize(text=p["target"], tokenizer=self.tokenizer, max_length=self.max_seq_len))
                   for p in self.train_pairs[task]]
        return [s for s in samples if len(s["input_ids"]) <= self.max_seq_len]
