import json
import os
import pathlib
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .base_trainer import BaseTrainer
from .training_config import TrainingConfig
from ..datasets import BaseDataset
from ..models import BaseModel
from ..tokenizers import BaseTokenizer


class CausalLmTrainer(BaseTrainer):
    TRAINER_STATE_FILENAME = "trainer_state.json"
    TRAINING_ARGS_FILENAME = "training_args.bin"

    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        args: TrainingConfig,
        train_dataset: BaseDataset,
        eval_dataset: BaseDataset | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
        )
        self.optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        os.makedirs(self.args.output_dir, exist_ok=True)

    def get_latest_checkpoint(self):
        files = os.listdir(self.args.output_dir)

        checkpoints = []
        for f in files:
            m = re.fullmatch(r"checkpoint-(\d+)", f)
            if m:
                checkpoints.append(int(m.group(1)))

        latest_checkpoint = max(checkpoints, default=0)
        return latest_checkpoint

    def resume_latest(self):
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint == 0:
            return 0, 0, 0
        else:
            foldername = f"checkpoint-{latest_checkpoint}"
            checkpoint_folder = self.args.output_dir / foldername
            return self.resume_from_folder(checkpoint_folder)

    def resume_from_folder(self, checkpoint_folder: pathlib.Path):
        self.model = self.model.__class__.from_pretrained(checkpoint_folder)

        trainer_state = self.load_trainer_state(checkpoint_folder)
        start_epoch = trainer_state["epoch"]
        start_batch_idx = trainer_state["batch_idx"]
        optimizer_steps = trainer_state["optimizer_steps"]
        return start_epoch, start_batch_idx, optimizer_steps

    def train(self, resume_from_checkpoint: os.PathLike[str] | bool = False):
        if resume_from_checkpoint:
            if isinstance(resume_from_checkpoint, bool):
                start_epoch, start_batch_idx, optimizer_steps = self.resume_latest()
            else:
                checkpoint_folder = pathlib.Path(resume_from_checkpoint).resolve()
                start_epoch, start_batch_idx, optimizer_steps = self.resume_from_folder(
                    checkpoint_folder
                )
        else:
            start_epoch, start_batch_idx, optimizer_steps = 0, 0, 0

        dataset_size = len(self.train_dataset)
        counter_width = len(str(dataset_size))

        self.model.train()
        total_loss = 0.0
        for epoch in range(start_epoch, self.args.num_train_epochs):
            print(f"EPOCH {epoch + 1}")
            print(f"=" * 75)

            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                if epoch <= start_epoch and batch_idx < start_batch_idx:
                    continue

                inputs = inputs.to(self.device)
                targets = inputs.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)

                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                loss = self.criterion(logits, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                optimizer_steps += 1

                total_loss += loss.item()

                if (
                    self.args.save_steps is not None
                    and optimizer_steps % self.args.save_steps == 0
                ):
                    self.save_state(epoch, batch_idx, optimizer_steps)

                    avg_loss = total_loss / self.args.train_batch_size
                    processed = (batch_idx + 1) * self.args.train_batch_size
                    batch_progress = f"[{processed:>{counter_width}d}/{dataset_size}]"

                    avg_loss_log = f"{avg_loss:.6f}"
                    steps_log = f"{optimizer_steps}"

                    total_loss = 0.0
                    self.log(
                        batch_progress, avg_loss=avg_loss_log, steps=optimizer_steps
                    )

    def load_trainer_state(self, checkpoint_folder: pathlib.Path) -> dict:
        json_file = checkpoint_folder / CausalLmTrainer.TRAINER_STATE_FILENAME
        with open(json_file, "r") as f:
            json_string = f.read()
            state_dict = json.loads(json_string)

        return state_dict

    def save_trainer_state(self, save_directory, epoch, batch_idx, optimizer_steps):
        state_dict = {
            "epoch": epoch,
            "batch_idx": batch_idx + 1,
            "optimizer_steps": optimizer_steps,
        }

        json_string = json.dumps(state_dict)
        save_path = save_directory / CausalLmTrainer.TRAINER_STATE_FILENAME

        with open(save_path, "w") as f:
            f.write(json_string)

    def save_state(self, epoch, batch_idx, optimizer_steps):
        checkpoint_folder = f"checkpoint-{optimizer_steps}"
        save_folder = self.args.output_dir / checkpoint_folder

        if os.path.exists(save_folder):
            raise ValueError(f"{save_folder} folder already exists")

        self.model.save_pretrained(save_directory=save_folder)
        self.save_trainer_state(save_folder, epoch, batch_idx, optimizer_steps)

        # optimizer.pt
        # scheduler.pt
        # training_args.bin
        # rng_state.pth

    def log(self, *args, **kwargs):
        args_string = " ".join(args)
        kwargs_string = " ".join([f"{k}={v}" for k, v in kwargs.items()])

        log_string = f"{args_string} {kwargs_string}"
        print(log_string)
