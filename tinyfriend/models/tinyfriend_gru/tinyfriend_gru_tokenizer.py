import os

import torch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents, Sequence
from tokenizers.trainers import BpeTrainer

from ...tokenizers import BaseTokenizer


class TinyFriendGruTokenizer(BaseTokenizer):
    tokenizer_json = "./data/tinyfriend_gru/tokenizer.json"

    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.normalizer = Sequence([NFD(), StripAccents()])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

        self.special_tokens = ["<unk>", "<eos>", "<pad>"]

        if os.path.exists(TinyFriendGruTokenizer.tokenizer_json):
            self.tokenizer = Tokenizer.from_file(TinyFriendGruTokenizer.tokenizer_json)

        self.unk_token = "<unk>"
        self.pad_token = "<eos>"
        self.eos_token = "<pad>"
        self.vocab = self.tokenizer.get_vocab()

    def _tokenize(self, text: str) -> list[str]:
        encoded = self.tokenizer.encode(text)
        return encoded.tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.decoder.decode(tokens)

    def train(
        self,
        dataset: Dataset,
        save_path: os.PathLike[str],
        vocab_size: int,
        min_frequency: int = 2,
    ):
        if os.path.exists(save_path):
            raise ValueError(f"{save_path} already exists")

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=self.special_tokens,
        )

        def batch_iterator(batch_size=1000):
            tok_dataset = dataset.select_columns("text")
            for batch in tok_dataset.iter(batch_size):
                yield batch["text"]

        self.tokenizer.train_from_iterator(
            batch_iterator(), trainer=trainer, length=len(dataset)
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.tokenizer.save(save_path)
