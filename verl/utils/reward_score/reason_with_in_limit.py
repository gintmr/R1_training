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

# Modified BY Xinrui Wu

import re
from typing import Dict

from mathruler.grader import extract_boxed_content, grade_answer


def limit_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def limit_acc_reward(predict_str: str, ground_truth: str) -> float:
    '''
    在当前设置下，使用二阶段推理依旧不影响：
    因为此函数的计算是基于整体答案，只需要不截断就好了
    '''
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def anwser_length_reward(current_length, budget) -> float:
    # stage = os.environ.get("stage", "1")
    # if stage == "1":
    #     target_length = budget + (budget // 50)
    # elif stage == "2":
    #     target_length = budget + (budget // 50) + 50 #g 参见verl/workers/rollout/vllm_rollout_spmd.py代码中padding_max_length = 值的计算过程
    
    target_length = budget
    anwser_length_reward = max((1 - 4 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0)
    
    return anwser_length_reward

def reason_with_in_limit_compute_score(predict_str: str, ground_truth: str, current_length:int, budget:int) -> Dict[str, float]:

    # print("current_length", current_length)
    # print("budget", budget)
    
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format = limit_format_reward(predict_str)
    accuracy = limit_acc_reward(predict_str, ground_truth)
    
    anwser_length = anwser_length_reward(current_length, budget)
    
    # print("format:", format, "accuracy:", accuracy, "anwser_length:", anwser_length)
    
    return {
        "overall": 0.75 * accuracy + 0.05 * format + 0.2 * anwser_length,
        "format": format,
        "accuracy": accuracy,
        "anwser_length": anwser_length
    }
