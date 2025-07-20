import re
import string

import torch

from .base_tokenizer import BaseTokenizer


class StrippedAsciiTokenizer(BaseTokenizer):
    def __init__(self):
        self.whitespace = " " + "\n"
        self.eos_token = "\x00"
        self.charset = (
            string.digits
            + string.ascii_letters
            + string.punctuation
            + self.whitespace
            + self.eos_token
        )
        self.vocab = dict(zip(self.charset, range(len(self.charset))))
        self.eos_token_id = self.vocab[self.eos_token]

        pattern = "|".join(re.escape(c) for c in self.charset)
        self.re = re.compile(pattern)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        tokens = self.re.findall(text)
        return tokens

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.charset[token_id]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)
