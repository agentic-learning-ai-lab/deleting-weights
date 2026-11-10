# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/google-deepmind/bbeh/blob/80d12ca916b7158f22293fcf3144f4d3d854d4be/bbeh/evaluate.py
# (reformatted, with docstrings added and the example prints and evaluate_correctness removed).

"""Answer scoring for BBEH (BIG-Bench Extra Hard).

Model predictions go through preprocess_sample, references through preprocess_reference, and the two are compared
with fuzzy_match.
"""


def strip_latex(response: str) -> str:
    """Strip a $...$, \\boxed{}, \\text{} or \\texttt{} wrapper from the response."""
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extract the final answer from the sample."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Return whether a preprocessed prediction matches a preprocessed reference."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    """Normalize a model prediction for fuzzy_match."""
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    """Normalize a reference answer for fuzzy_match."""
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference
