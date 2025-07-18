import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tinybard_char_rnn_config import TinyBardCharRnnConfig


class TinyBardCharRnnModule(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_size, num_layers, dropout: float = 0.0
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_ids, hidden=None):
        inputs = self.embed(input_ids)
        outputs, hidden = self.rnn(inputs, hidden)
        logits = self.fc(outputs)
        return logits, hidden


class TinyBardCharRnnModel(nn.Module):
    def __init__(self, config: TinyBardCharRnnConfig):
        super().__init__()

        self.config = config
        self.module = TinyBardCharRnnModule(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        assert (
            input_ids.dtype == torch.int64
        ), f"Expected input_ids dtype to be torch.int64, got: {input_ids.dtype}"

        return self.module(input_ids, hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        eos_token_id: int | None = None,
        streamer=None,
    ):
        generated = [input_ids]
        hidden = None

        if streamer is not None:
            streamer.put(input_ids)

        model_input = input_ids
        for _ in range(max_new_tokens):
            logits, hidden = self(model_input, hidden)
            logits = logits[-1, :]

            if temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            model_input = next_token
            hidden = hidden.detach()

            if eos_token_id is not None and next_token == eos_token_id:
                break

            if streamer is not None:
                streamer.put(next_token)

        if streamer is not None:
            streamer.end()

        return torch.cat(generated)

    @staticmethod
    def from_pretrained(
        pretrained_model_path: str | os.PathLike, device_map: str = "auto"
    ):
        if device_map == "auto":
            device_map = "cuda" if torch.cuda.is_available() else "cpu"

        model_folder = pathlib.Path(pretrained_model_path).resolve()
        model_file = model_folder / "model.pth"

        config = TinyBardCharRnnConfig.from_pretrained(model_folder)
        model = TinyBardCharRnnModel(config)

        model_state_dict = torch.load(model_file, map_location=device_map)
        model.load_state_dict(model_state_dict)
        model.to(device_map)

        return model

    def save_pretrained(self, save_directory: str | os.PathLike):
        save_folder = pathlib.Path(save_directory).resolve()
        model_path = save_folder / "model.pth"

        os.makedirs(save_folder, exist_ok=True)

        torch.save(self.state_dict(), model_path)

        self.config.save_pretrained(save_folder)
