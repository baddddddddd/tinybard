import pathlib

import torch
from torch.utils.data import Dataset


class TheEchoChamberDataset(Dataset):
    def __init__(self, seq_len: int, tokenizer):
        module_dir = pathlib.Path(__file__).resolve().parent
        raw_dataset = module_dir / "raw.txt"

        with open(raw_dataset, "r") as f:
            self.raw_text = f.read()

        self.seq_len = seq_len
        self.encoded = tokenizer.encode(self.raw_text)

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        x = torch.LongTensor(self.encoded[idx : idx + self.seq_len])
        y = torch.LongTensor(self.encoded[idx + 1 : idx + 1 + self.seq_len])

        return x, y
