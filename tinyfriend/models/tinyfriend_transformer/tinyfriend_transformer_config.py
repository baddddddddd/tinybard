from ..base_config import BaseConfig


class TinyFriendTransformerConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
