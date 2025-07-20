import re
import string
import unicodedata

from .base_tokenizer import BaseTokenizer


class StrippedAsciiTokenizer(BaseTokenizer):
    def __init__(self):
        self.whitespace = " \n\t"
        self.eos_token = "\x00"
        self.pad_token = "\x01"
        self.charset = (
            string.digits
            + string.ascii_letters
            + string.punctuation
            + self.whitespace
            + self.eos_token
            + self.pad_token
        )
        self.vocab = dict(zip(self.charset, range(len(self.charset))))

        pattern = "|".join(re.escape(c) for c in self.charset)
        self.re = re.compile(pattern)

    def _normalize_to_ascii(self, text: str) -> str:
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        normalized = self._normalize_to_ascii(text)
        tokens = self.re.findall(normalized)
        return tokens

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.charset[token_id]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)
