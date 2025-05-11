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
import json
import os
from mathruler.grader import extract_boxed_content, grade_answer


def limit_format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)

    pattern = re.compile(
        r"(?s)"  # 启用 DOTALL 模式，让 . 匹配包括换行符在内的任意字符
        r"<think>.*?</think>"  # 匹配 <think> 和 </think> 之间的任意内容，包括换行符
        r"\n?"  # 匹配可能出现的换行符，可以出现 0 次或 1 次
        r"\*\*Final Answer\*\*\\boxed\{.*\}.*?"  # 匹配 **Final Answer**\boxed{数字}
    )
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

    # if budget >= 800:
    #     if current_length >= budget:
    #         target_length = budget + budget // 8 #g 将阈值适当左移
    #         anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
    #         #g 更加严格的惩罚，超过1/4*budget就是0
    #     else:
    #         anwser_length_reward = 1.0
    # else:
    #     target_length = budget - budget // 8 #g 将阈值适当左移
    #     if current_length <= target_length:
    #         target_length = budget
    #         anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
    #         #g 更加严格的惩罚，超过1/4*budget就是0
    #     else:
    #         anwser_length_reward = 1.0
    # import os
    # if os.environ['steady'] == "FULLv5" or os.environ['steady'] == "FULLv6":
    target_length = budget
    if budget >= 800:
        if current_length >= budget:
            anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
            #g 更加严格的惩罚，超过1/4*budget就是0
        else:
            anwser_length_reward = max((1 - ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
    else:
        if current_length >= budget:
            anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
            #g 更加严格的惩罚，超过1/4*budget就是0
        else:
            anwser_length_reward = max((1 - ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2

    return anwser_length_reward

def reason_with_in_limit_compute_score(predict_str: str, ground_truth: str, current_length:int, budget:int, current_epoch:int, prompt_str:str, raw_response_str:str) -> Dict[str, float]:

    # print("current_length", current_length)
    # print("budget", budget)

    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format = limit_format_reward(predict_str)
    accuracy = limit_acc_reward(predict_str, ground_truth)

    anwser_length = anwser_length_reward(current_length, budget)

    # print("format:", format, "accuracy:", accuracy, "anwser_length:", anwser_length)
    # if accuracy == 1.0:
    #     anwser_length = 1.0
    #     format = 1.0
    #g -------
    # 动态调整权重
    # if accuracy < 1.0:  # 如果答案错误
    #     # 增加长度奖励的权重，鼓励更长的输出
    #     length_weight = 0.3  # 比原来的0.075/0.125更高
    #     format_weight = 0.1
    #     accuracy_weight = 0.6
    # else:
    #     # 正确答案保持原有权重
    #     length_weight = 0.075
    #     format_weight = 0.075
    #     accuracy_weight = 0.85
    #g ---------
    # length_weight = 0.075
    # format_weight = 0.075
    # accuracy_weight = 0.85
    length_weight = 0.1
    format_weight = 0.225
    accuracy_weight = 0.675
    # import os
    # # if budget <= 800 and os.environ['steady']=='FULLv9':
    # if budget <= 800:
    #     length_weight = 0.075
    #     format_weight = 0.225
    #     accuracy_weight = 0.7
    # overall = 0.85 * accuracy + 0.075 * format + 0.075 * anwser_length
    overall = accuracy_weight * accuracy + format_weight * format + length_weight * anwser_length
    # overall = 0.825 * accuracy + 0.025 * format + 0.15 * anwser_length
    # if current_epoch == 1:
    #     overall = 0.825 * accuracy + 0.025 * format + 0.15 * anwser_length
    # elif current_epoch == 2:
    #     overall = 0.825 * accuracy + 0.025 * format + 0.15 * anwser_length
    # else:
    #     overall = 0.85 * accuracy + 0.025 * format + 0.125 * anwser_length
    
    
    if accuracy == 1 and format == 1 and current_length >= 500 and current_length <= 2500:
        file_path = f"QA_pairs/qa_pairs_{current_epoch}.jsonl"
        qa_pair = {"prompt": prompt_str, "response": raw_response_str}
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(qa_pair, f, ensure_ascii=False)
            f.write("\n")
    
    return {
        # 0.9 accuracy + 0.1 format
        # "overall": 0.75 * accuracy + 0.05 * format + 0.2 * anwser_length,
        "overall": overall,
        "format": format,
        "accuracy": accuracy,
        "anwser_length": anwser_length
    }
