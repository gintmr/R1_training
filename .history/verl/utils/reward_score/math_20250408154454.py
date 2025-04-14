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

import re

from mathruler.grader import extract_boxed_content, grade_answer


def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def math_budget_reward(budget, response_len):
    if response_len <= budget:
        return 0
    else:
        return -1 * (response_len - budget) / (2 + response_len - budget)

def math_compute_score(predict_str: str, ground_truth: str) -> float:  # , budget: int, response_len: int) -> float:
    # For our method, we manually add <think> and </think> to the answer, so no need to add format reward
    return math_acc_reward(predict_str, ground_truth)
    # return 0.9 * math_acc_reward(predict_str, ground_truth) + 0.1 * math_format_reward(predict_str) # + math_budget_reward(budget, response_len)
    