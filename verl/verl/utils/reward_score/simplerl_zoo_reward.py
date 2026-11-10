# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SimpleRL-Zoo reward, adapted from https://github.com/hkust-nlp/simpleRL-reason at commit cf1c785
# (verl/utils/reward_score/hf_math_verify.py): trimmed to the default reward, with docstrings and type hints added.
import gc
import logging
import re
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Any, Callable, Optional

from math_verify import parse, verify

from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer


class GlobalProcessPool:
    """Process pool shared by every reward call in this process, recreated if it breaks."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int):
        """Create the pool."""
        self.max_workers = max_workers
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()

    def _initialize_executor(self) -> None:
        """Start a new process pool, shutting down the old one."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
            gc.collect()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    @classmethod
    def get_instance(cls, max_workers: int) -> "GlobalProcessPool":
        """Return the shared pool, creating it on first use."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers)
        return cls._instance

    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit a task to the executor, recreating the pool if it is broken."""
        try:
            if self.executor is None:
                with self._lock:
                    self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            with self._lock:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)


global_executor = GlobalProcessPool.get_instance(max_workers=2)
VERIFY_TIMEOUT_SECONDS = 10


def extract_solution(solution_str: str) -> str:
    """Return the model's final answer."""
    model_output = re.sub(r"^.*?<\|im_start\|>assistant", "<|im_start|>assistant", solution_str, flags=re.DOTALL,
                          count=1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    return qwen_extract_answer(model_output, data_name="math")


def hf_verify_with_try(gold: str, target: str) -> bool:
    """Return whether math_verify judges the target equal to the gold answer (False on a parsing error)."""
    try:
        parsed_target = parse(target)
        parsed_gold = parse(gold)
        return verify(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def hf_math_equal_subprocess(gold: str, target: str) -> bool:
    """Run hf_verify_with_try in the process pool with a timeout; a timeout or error counts as incorrect."""
    try:
        future = global_executor.submit(hf_verify_with_try, gold=gold, target=target)
        return future.result(timeout=VERIFY_TIMEOUT_SECONDS)
    except TimeoutError:
        print(f"Timeout occurred for gold {gold} and target {target}.")
        return False
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Return the SimpleRL-Zoo reward: 1 if the final answer equals the ground truth (math_verify), else 0."""
    extract_answer = extract_solution(solution_str=solution_str)
    boxed_answer = extract_answer if "\\boxed" in extract_answer else f"\\boxed{{{extract_answer}}}"
    boxed_ground_truth = ground_truth if "\\boxed" in ground_truth else f"\\boxed{{{ground_truth}}}"
    correct = hf_math_equal_subprocess(gold=boxed_ground_truth, target=boxed_answer)
    return float(correct)
