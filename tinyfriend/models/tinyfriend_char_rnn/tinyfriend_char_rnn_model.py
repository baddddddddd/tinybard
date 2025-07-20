import torch
import torch.nn as nn

from ..base_model import BaseModel
from .tinyfriend_char_rnn_config import TinyFriendCharRnnConfig


class TinyFriendCharRnnModel(BaseModel):
    def __init__(self, config: TinyFriendCharRnnConfig):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
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
