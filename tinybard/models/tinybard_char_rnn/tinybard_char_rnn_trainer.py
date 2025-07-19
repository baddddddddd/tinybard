import json
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
        tokenizer: TinyBardCharRnnTokenizer,
        args: TinyBardCharRnnTrainingArguments,
        train_dataset: Dataset | IterableDataset,
        eval_dataset: Dataset | IterableDataset | None = None,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(args.output_dir, exist_ok=True)

    def train(self, resume_from_checkpoint: str | os.PathLike | None = None):
        if resume_from_checkpoint:
            checkpoint_folder = pathlib.Path(resume_from_checkpoint).resolve()
            self.model = TinyBardCharRnnModel.from_pretrained(checkpoint_folder).to(
                self.device
            )

            optimizer_file = checkpoint_folder / "optimizer.pth"
            self.optimizer.load_state_dict(
                torch.load(optimizer_file, map_location=self.device)
            )

            trainer_state_file = checkpoint_folder / "trainer_state.json"
            with open(trainer_state_file, "r") as f:
                json_string = f.read()
                trainer_state_dict = json.loads(json_string)

            start_epoch = trainer_state_dict["epoch_idx"]
            step_count = trainer_state_dict["optimizer_steps"]
        else:
            start_epoch = 0
            step_count = 0

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch_idx in range(start_epoch, self.args.num_train_epochs):
            print(f"EPOCH {epoch_idx + 1}")
            print("=" * 50)

            total_loss = 0
            loss_steps = 0
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

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                step_count += 1

                total_loss += loss.item()
                loss_steps += 1

                if self.args.save_steps > 0 and (
                    step_count % self.args.save_steps == 0
                    or (batch_idx + 1) == len(dataloader)
                ):
                    n = len(self.train_dataset)
                    cur = min((batch_idx + 1) * self.args.train_batch_size, n)
                    width = len(str(n))

                    avg_loss = total_loss / loss_steps
                    loss_steps = 0
                    total_loss = 0

                    print(f"[{cur:>{width}d}/{n}] avg_loss={avg_loss:.6f}")

                    self.save_checkpoint(step_count, epoch_idx)

    def save_checkpoint(self, step, epoch_idx):
        folder_name = f"checkpoint-{step}"
        folder_path = self.args.output_dir / folder_name

        os.makedirs(folder_path, exist_ok=True)

        self.model.save_pretrained(folder_path)

        optimizer_file = folder_path / "optimizer.pth"
        torch.save(self.optimizer.state_dict(), optimizer_file)

        trainer_state_dict = {
            "optimizer_steps": step,
            "epoch_idx": epoch_idx,
        }

        json_string = json.dumps(trainer_state_dict)
        trainer_state_file = folder_path / "trainer_state.json"
        with open(trainer_state_file, "w") as f:
            f.write(json_string)
