import re
import string

import torch


class BaseTokenizer:
    def __init__(self):
        self.vocab = {}
        self.unk_token = ""
        self.eos_token = ""
        self.pad_token = ""

    @property
    def unk_token_id(self):
        return self._convert_token_to_id(self.unk_token)

    @property
    def eos_token_id(self):
        return self._convert_token_to_id(self.eos_token)

    @property
    def pad_token_id(self):
        return self._convert_token_to_id(self.pad_token)

    def __call__(
        self, texts: list[str], return_tensors: str | None = None
    ) -> list[list[int] | torch.Tensor]:
        encoded = [self.encode(text, return_tensors=return_tensors) for text in texts]
        return encoded

    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("_tokenize() method is not implemented")

    def _convert_token_to_id(self, token: str) -> int:
        raise NotImplementedError("_convert_token_to_id() method is not implemented")

    def _convert_id_to_token(self, token_id: int) -> str:
        raise NotImplementedError("_convert_id_to_token() method is not implemented")

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError(
            "_convert_tokens_to_string() method is not implemented"
        )

    def tokenize(self, text: str) -> list[str]:
        return self._tokenize(text)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = [self._convert_token_to_id(token) for token in tokens]
        return ids

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        tokens = [self._convert_id_to_token(id_) for id_ in ids]
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._convert_tokens_to_string(tokens)

    def encode(
        self, text: str, return_tensors: str | None = None
    ) -> list[int] | torch.Tensor:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)

        if return_tensors == "pt":
            return torch.LongTensor(ids)
        else:
            return ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        s = self.convert_tokens_to_string(tokens)
        return s

    def batch_decode(self, sequences: list[list[int] | torch.Tensor]) -> list[str]:
        decoded = [self.decode(seq) for seq in sequences]
        return decoded

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)
