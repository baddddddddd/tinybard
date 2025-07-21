import torch
import torch.nn as nn
import torch.nn.functional as F

from .tinyfriend_rnn_config import TinyFriendRnnConfig
from ..base_model import BaseModel
from ...utils import BaseStreamer, top_p_sample


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
        self.config: TinyFriendRnnConfig

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.module(input_ids)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        streamer: BaseStreamer | None = None,
    ):
        generated = [input_ids]
        hidden = None

        if streamer is not None:
            streamer.put(input_ids)

        model_input = input_ids
        for _ in range(max_new_tokens):
            logits, hidden = self.module(model_input, hidden)
            logits = logits[-1, :]

            if temperature > 0.0:
                logits /= temperature
                if top_p > 0.0:
                    next_token = top_p_sample(logits, p=top_p)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            model_input = next_token.detach()
            hidden = hidden.detach()

            if (
                self.config.eos_token_id is not None
                and next_token.item() == self.config.eos_token_id
            ):
                break

            if streamer is not None:
                streamer.put(next_token)

        if streamer is not None:
            streamer.end()

        return torch.cat(generated)
