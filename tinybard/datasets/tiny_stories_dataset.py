import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, seq_len, stride):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

        def encode(row):
            text = row["text"] + tokenizer.eos_token
            token_ids = tokenizer.encode(text)
            return {"input_ids": token_ids}

        num_proc = os.cpu_count() - 1
        self.dataset = raw_dataset.map(
            encode, remove_columns=["text"], num_proc=num_proc
        )

        self.idx_to_doc = []
        for row_idx, row in enumerate(self.dataset):
            token_ids = row["input_ids"]
            for start_idx in range(0, len(token_ids) - seq_len, stride):
                doc = (row_idx, start_idx)
                self.idx_to_doc.append(doc)

    def __len__(self):
        return len(self.idx_to_doc)

    def __getitem__(self, idx):
        row_idx, start_idx = self.idx_to_doc[idx]
        row = self.dataset[row_idx]
        token_ids = row["input_ids"]

        inputs = torch.LongTensor(token_ids[start_idx : start_idx + self.seq_len])
        targets = torch.LongTensor(
            token_ids[start_idx + 1 : start_idx + 1 + self.seq_len]
        )

        return inputs, targets
