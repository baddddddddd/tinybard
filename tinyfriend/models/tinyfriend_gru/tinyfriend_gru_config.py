from ..base_config import BaseConfig


class TinyFriendGruConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
