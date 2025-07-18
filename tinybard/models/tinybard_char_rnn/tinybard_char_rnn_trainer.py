import pathlib
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
        save_steps: int = 0,
    ):
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.weight_decay = weight_decay
        self.save_steps = save_steps


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

        os.makedirs(args.output_dir, exist_ok=True)

    def train(self):
        print(f"Using device: {self.device}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
        )

        self.model.train()
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

                if (
                    self.args.save_steps > 0
                    and (batch_idx + 1) % self.args.save_steps == 0
                ):
                    n = len(self.train_dataset)
                    cur = (batch_idx + 1) * self.args.train_batch_size
                    width = len(str(n))
                    print(f"[{cur:>{width}d}/{n}] loss={loss.item():.6f}")

                    self.save_checkpoint(batch_idx + 1)

    def save_checkpoint(self, step):
        folder_name = f"checkpoint-{step}"
        folder_path = self.args.output_dir / folder_name

        os.makedirs(folder_path, exist_ok=False)

        model_path = folder_path / "model.pth"
        optimizer_path = folder_path / "checkpoint.pth"

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
