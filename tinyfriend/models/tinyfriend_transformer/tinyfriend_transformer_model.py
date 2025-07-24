import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from .tinyfriend_transformer_config import TinyFriendTransformerConfig
from ..base_model import BaseModel


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal=False
):
    kdim = key.size(-1)
    seq_len = key.size(-2)
    scale_factor = 1 / math.sqrt(kdim)

    attention_scores = query @ key.transpose(-2, -1) * scale_factor

    if is_causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    attention_weights = torch.softmax(attention_scores, dim=-1)
    output = attention_weights @ value

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ):
        batch_size, seq_len, _ = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        values = values.transpose(1, 2)
        values = values.reshape(batch_size, seq_len, self.embed_dim)
        outputs = self.out_proj(values)

        return outputs, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim: int, intermediate_size: int):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=intermediate_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=intermediate_size, out_features=embed_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.self_attention = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.attention_dropout = nn.Dropout(p=dropout)
        self.post_attention_norm = nn.LayerNorm(d_model)

        self.feed_forward = PositionWiseFeedForward(
            embed_dim=d_model, intermediate_size=dim_feedforward
        )
        self.ff_dropout = nn.Dropout(p=dropout)
        self.post_ff_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output, _ = self.self_attention(x, is_causal=True)
        x = self.post_attention_norm(x + self.attention_dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.post_ff_norm(x + self.ff_dropout(ff_output))

        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.stack = nn.Sequential(
            *[
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.stack(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int, dropout: float):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_dim).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, 2 * (i // 2) / embed_dim)
        encoding = pos * angle_rates

        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        self.register_buffer("pe", encoding.unsqueeze(0))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TinyFriendTransformer(nn.Module):
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
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim, max_len=max_len, dropout=dropout
        )
        self.transformer = TransformerStack(
            num_layers=num_layers,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.embedding_scale_factor = math.sqrt(embed_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids) * self.embedding_scale_factor
        x = self.positional_encoding(x)
        x = self.transformer(x)

        logits = x @ self.embedding.weight.T
        return logits


class TinyFriendTransformerModel(BaseModel):
    config_class = TinyFriendTransformerConfig

    def __init__(self, config: TinyFriendTransformerConfig):
        super().__init__(config)
        self.config: TinyFriendTransformerConfig
        self.module = TinyFriendTransformer(
            vocab_size=config.vocab_size,
            max_len=config.max_len,
            embed_dim=config.embed_dim,
            num_layers=config.num_layer,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
        )

    def forward(self, input_ids: torch.Tensor):
        return self.module(input_ids)
