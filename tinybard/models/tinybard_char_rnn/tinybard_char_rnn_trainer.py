import os

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader

from .tinybard_char_rnn_model import TinyBardCharRnnModel
from .tinybard_char_rnn_tokenizer import TinyBardCharRnnTokenizer


class TinyBardCharRnnTrainingArguments:
    def __init__(
        self,
        output_dir: str,
        learning_rate: float,
        num_train_epochs: int,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        weight_decay: float = 1e-5,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.weight_decay = weight_decay


class TinyBardCharRnnTrainer:
    def __init__(
        self,
        model: TinyBardCharRnnModel,
        args: TinyBardCharRnnTrainingArguments,
        train_dataset: Dataset | IterableDataset,
        processing_class: TinyBardCharRnnTokenizer,
        eval_dataset: Dataset | IterableDataset | None = None,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = model.to(self.device)
        self.args = args
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print(f"Using device: {self.device}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
        )

        for epoch_idx in range(self.args.num_train_epochs):
            print(f"EPOCH {epoch_idx + 1}")
            print("=" * 50)

            for batch_idx, sample in enumerate(dataloader):
                inputs, targets = sample
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(inputs)

                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                loss = self.criterion(logits, targets)
                loss.backward()

                self.optimizer.step()

                if batch_idx % 10 == 0:
                    n = len(self.train_dataset)
                    cur = (batch_idx + 1) * self.args.train_batch_size
                    width = len(str(n))
                    print(f"[{cur:>{width}d}/{n}] loss={loss.item():.6f}")
