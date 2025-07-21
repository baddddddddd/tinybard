import torch
import torch.nn as nn

from ..base_model import BaseModel
from .tinyfriend_rnn_config import TinyFriendRnnConfig


class TinyFriendRnnModule(nn.Module):
    def __init__(self, config: TinyFriendRnnConfig):
        super().__init__()
        if config.pad_token_id is None:
            self.embed = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
            )
        else:
            self.embed = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=config.pad_token_id,
            )

        self.rnn: nn.RNN | nn.GRU | nn.LSTM
        if config.architecture == "vanilla":
            self.rnn = nn.RNN(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "gru":
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"{config.architecture} is an unknown RNN architecture")

        self.fc = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size
        )

    def forward(self, input_ids, hidden=None):
        inputs = self.embed(input_ids)
        outputs, hidden = self.rnn(inputs, hidden)
        logits = self.fc(outputs)
        return logits, hidden


class TinyFriendRnnModel(BaseModel):
    config_class = TinyFriendRnnConfig

    def __init__(self, config: TinyFriendRnnConfig):
        super().__init__(config)
        self.module = TinyFriendRnnModule(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.module(input_ids)
        return logits
