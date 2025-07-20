import os

import datasets
import torch

from ..tokenizers import BaseTokenizer


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: BaseTokenizer,
        chunk_size: int,
        stride: int | None = None,
    ):
        if stride is None:
            stride = chunk_size

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride

        raw_dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)

        def encode(example):
            sequences = [text + tokenizer.eos_token for text in example["text"]]
            encoded = tokenizer(sequences)
            for i in range(len(encoded)):
                rem = (len(encoded[i]) - chunk_size) % stride
                if rem > 1:
                    pad_size = chunk_size - rem + 1
                    encoded[i] += [tokenizer.pad_token_id] * pad_size

            return {"input_ids": encoded}

        num_proc = os.cpu_count()
        self.dataset = raw_dataset.map(
            encode,
            remove_columns=["text"],
            batched=True,
            num_proc=num_proc,
        )

        self.idx_to_doc = []
        for row_idx in range(len(self.dataset)):
            token_ids = self.dataset[row_idx]["input_ids"]
            for start_idx in range(0, len(token_ids) - chunk_size, stride):
                doc = (row_idx, start_idx)
                self.idx_to_doc.append(doc)

    def __len__(self):
        return len(self.idx_to_doc)

    def __getitem__(self, idx):
        row_idx, start_idx = self.idx_to_doc[idx]
        row = self.dataset[row_idx]
        token_ids = row["input_ids"]

        inputs = torch.LongTensor(token_ids[start_idx : start_idx + self.chunk_size])
        targets = torch.LongTensor(
            token_ids[start_idx + 1 : start_idx + 1 + self.chunk_size]
        )

        return inputs, targets
