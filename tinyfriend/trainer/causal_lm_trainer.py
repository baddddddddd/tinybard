import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .base_trainer import BaseTrainer
from .training_config import TrainingConfig
from ..models import BaseModel
from ..tokenizers import BaseTokenizer


class CausalLmTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        args: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self.device}")

        self.model = model.to(device=self.device)
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

    def train(self, resume_from_checkpoint: str | bool = False):
        self.model.train()

        for epoch in range(self.args.num_train_epochs):
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
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
