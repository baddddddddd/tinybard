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
    ):
        self.architecture = architecture
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
