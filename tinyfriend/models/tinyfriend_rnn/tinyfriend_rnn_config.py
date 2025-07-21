from ..base_config import BaseConfig


class TinyFriendRnnConfig(BaseConfig):
    def __init__(
        self,
        architecture: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        self.architecture = architecture
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
