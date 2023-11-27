# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import os
import json

import torch
from torch.utils.data import Dataset


# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=512):
        print("INSIDE INIT FUNCTION for partition: ", partition)

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.partition_use = partition
        self.max_words_in_dataset = 0
        
        ###########################
        path = "/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/bbq_train_dataset.json"
        if 'training_data_path' in os.environ:
            path = os.environ['training_data_path']
        print("TRAIN PATH is: ", path)
        self.ann = json.load(open(path))
        print("Initial len is: ", len(self.ann))
        self.ann = list(filter(lambda x: self.is_input_valid(x), self.ann))
        print("Final len is: ", len(self.ann))

        
        ########################
        
        FRAC_TO_TRANSFER  = 0.8
        IDX_END_TRAIN = int(FRAC_TO_TRANSFER * len(self.ann))

        if partition == "train":
            self.ann = self.ann[:IDX_END_TRAIN]
        else:
            self.ann = self.ann[IDX_END_TRAIN:]
            self.ann = self.ann[:100]
        ##############################
        
        print("MAX WORDS in dataset is: ", self.max_words_in_dataset)
        if self.max_words_in_dataset < self.max_words:
            self.max_words = self.max_words_in_dataset

        

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        # print("INDEX: ", index, " for partition: ", self.partition_use)

        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        prompt = ann['instruction']
        example = prompt + ann["output"]
        # print("Example is: ", example)
        # print("########")
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        # print("Padding amt: ", padding)
        if padding > 0:
            example = torch.cat(
                (example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
            # print("Having to truncate")
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

    def is_input_valid(self, elem):

        ann = elem
        prompt = ann['instruction']
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        
        self.max_words_in_dataset = max(self.max_words_in_dataset, example.shape[0]+5)
        
        if padding < 0:
            return False
        return True
