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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import random

class CurriculumCollator:
    def __init__(self, total_epoches):
        self.total_epoches = total_epoches
        self.current_epoch = 1  # 需在训练循环中更新

    def get_progress(self):
        progress = self.current_epoch / self.total_epoches
        self.current_epoch += 1
        return progress

    def get_budget(self, progress):
        if progress <= 0.2:
            budget_list = [4800, 5600, 6400]
        elif progress <= 0.4 and progress > 0.2:
            budget_list = [3200, 4000, 4800]
        elif progress <= 0.6 and progress > 0.4:
            budget_list = [800, 1600, 2400, 3200]
        elif progress <= 0.8 and progress > 0.6:
            budget_list = [100, 200, 400, 800, 1600, 3200]
        else:
            budget_list = [100, 200, 400, 800, 1600, 3200, 4800, 5600, 6400]
        return random.choice(budget_list)
    
    def __call__(self, features: List[Dict[str, Any]]):
        progress = self.get_progress()
        print("#" * 50 + f"Current Progress: {progress:.2f}" + "#" * 50)
        budget = self.get_budget(progress)
        return collate_fn(features, budget)


def collate_fn(features: List[Dict[str, Any]], budget) -> Dict[str, Any]:
    
    # # Create budget ranges for curriculum learning
    # all_budgets = [50*2**i for i in range(1, 8)]   # 100-6400 tokens
    # # Select one random budget for the entire batch
    # budget = random.choice(all_budgets)
    #g 👆随机选取

    budget_and_tokens = budget + (budget // 50)
    print(f"budget_and_tokens = {budget_and_tokens}")
    # Get tokenizer from the dataset class instead of individual features
    tokenizer = features[0]["dataset"].tokenizer
    
    for feature in features:
        # Add budget tag to prompt
        prompt = feature["prompt_txt"]
        prompt_list = prompt.split("<｜Assistant｜>")
        assert budget % 50 == 0, "budget must be a multiple of 50"
        if len(prompt_list) == 1:
            print(f"Warning: prompt {prompt} has no assistant segment, the budget tag will be added to the first segment")
            prompt = prompt_list[0] + f"\n(Complete thinking within \n<remaining>{budget}</remaining>\n tokens or fewer.)"
            
        elif len(prompt_list) == 2: 
            # Add budget tag
            prompt = prompt_list[0] + f"\n(Complete thinking within \n<remaining>{budget}</remaining>\n tokens or fewer.)" + "<｜Assistant｜>" + prompt_list[1]
            
        else:
            print(f"Warning: prompt {prompt} has more than two segments, only the first two segments will be tagged with budget")
            prompt = prompt_list[0] + f"\n(Complete thinking within \n<remaining>{budget}</remaining>\n tokens or fewer.)" + "<｜Assistant｜>" + prompt_list[1]
            
        feature['prompt_txt'] = prompt
        new_raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if len(new_raw_prompt_ids) > feature["input_ids"].shape[-1]:
            print("*"*50, f"Warning: after adding the budget, the prompt is longer than the max token budget, the new prompt is {len(new_raw_prompt_ids)}, but the max budget is {feature['input_ids'].shape[-1]}", "*"*50)
            print(prompt)
            feature["raw_prompt_ids"] = new_raw_prompt_ids
            # Create new attention_mask (0s for padding, 1s for content)
            feature["attention_mask"] = torch.tensor(
                [1] * feature["input_ids"].shape[-1]
            )
            # Create new position_ids (0s for padding, then 0,1,2... for content)
            feature["position_ids"] = torch.tensor(
                list(range(feature["input_ids"].shape[-1]))
            )
            # Pad input_ids to match max_length
            feature["input_ids"] = torch.tensor(
                new_raw_prompt_ids[-feature["input_ids"].shape[-1]:]
            )
        else: 
            max_length = feature["input_ids"].shape[-1]
            
            # Create padded raw_prompt_ids (pad at beginning)
            pad_length = max_length - len(new_raw_prompt_ids)
            feature["raw_prompt_ids"] = new_raw_prompt_ids
            
            # Create new attention_mask (0s for padding, 1s for content)
            feature["attention_mask"] = torch.tensor(
                [0] * pad_length + [1] * len(new_raw_prompt_ids)
            )
            
            # Create new position_ids (0s for padding, then 0,1,2... for content)
            feature["position_ids"] = torch.tensor(
                [0] * pad_length + list(range(len(new_raw_prompt_ids)))
            )
            
            # Pad input_ids to match max_length
            feature["input_ids"] = torch.tensor(
                [tokenizer.pad_token_id] * pad_length + new_raw_prompt_ids
            )
        
        # Remove dataset reference to avoid memory issues
        if "dataset" in feature:
            del feature["dataset"]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        prompt_str: str = row_dict[self.prompt_key]
        if self.format_prompt:
            prompt_str = prompt_str + " " + self.format_prompt.strip()

        if self.image_key in row_dict:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_data"] = {"image": images}
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            messages = [{"role": "user", "content": prompt_str}]
            messages.insert(0, {"role": "system", "content": "Return your final response within \\boxed{}. "})
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        row_dict["dataset"] = self
        row_dict['prompt_txt'] = prompt
        return row_dict
