from __future__ import annotations

import json
import os
import pathlib


class TinyBardCharRnnConfig:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    @staticmethod
    def from_pretrained(
        pretrained_model_path: str | os.PathLike,
    ) -> TinyBardCharRnnConfig:
        model_path = pathlib.Path(pretrained_model_path).resolve()
        config_filename = "config.json"
        config_filepath = model_path / config_filename

        return TinyBardCharRnnConfig.from_json_file(config_filepath)

    @staticmethod
    def from_dict(config_dict: dict) -> TinyBardCharRnnConfig:
        return TinyBardCharRnnConfig(
            vocab_size=config_dict["vocab_size"],
            embedding_dim=config_dict["embedding_dim"],
            hidden_size=config_dict["hidden_size"],
            num_layers=config_dict["num_layers"],
            dropout=config_dict["dropout"],
        )

    @staticmethod
    def from_json_file(json_file: str | os.PathLike) -> TinyBardCharRnnConfig:
        with open(json_file, "r") as f:
            content = f.read()
            config_dict = json.loads(content)

        return TinyBardCharRnnConfig.from_dict(config_dict)

    def save_pretrained(self, save_directory):
        save_folder = pathlib.Path(save_directory).resolve()

        os.makedirs(save_folder, exist_ok=True)

        json_file_path = save_folder / "config.json"
        self.to_json_file(json_file_path)

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def to_json_file(self, json_file_path: str | os.PathLike):
        with open(json_file_path, "w") as f:
            json_string = self.to_json_string()
            f.write(json_string)

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        json_string = json.dumps(config_dict)
        return json_string
