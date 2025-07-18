import re
import string


class StrippedAsciiTokenizer:
    def __init__(self):
        self.whitespace = " " + "\n"
        self.charset = (
            string.digits + string.ascii_letters + string.punctuation + self.whitespace
        )
        self.vocab = dict(zip(self.charset, range(len(self.charset))))

        pattern = "|".join(re.escape(c) for c in self.charset)
        self.re = re.compile(pattern)

    def tokenize(self, text: str, **kwargs) -> list[str]:
        tokens = self.re.findall(text)
        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = list(map(lambda token: self.vocab[token], tokens))
        return ids

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        tokens = list(map(lambda id: self.charset[id], ids))
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        s = "".join(tokens)
        return s

    def decode(self, token_ids: list[int]) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        s = self.convert_tokens_to_string(tokens)
        return s

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)
