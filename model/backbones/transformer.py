"""
Adapted from @seitalab
Source: https://github.com/seitalab/dnn_ecg_comparison/blob/main/experiment/codes/architectures/transformer.py

"""
import math
from typing import Any

import torch
import torch.nn as nn

clamp_val = 20

class LinearEmbed(nn.Module):

    def __init__(self, num_lead: int, chunk_len: int, d_model: int) -> None:
        super(LinearEmbed, self).__init__()

        self.num_lead = num_lead
        self.chunk_len = chunk_len

        chunk_dim = self.num_lead * self.chunk_len
        self.embed = nn.Linear(chunk_dim, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, seqlen).
        Returns:
            feat (torch.Tensor): Tensor of size (batch_size, num_chunks, d_model).
        """
        assert(x.size(1) == self.num_lead)
        assert(x.size(2) % self.chunk_len == 0)

        bs = x.size(0)
        num_chunks = x.size(2) // self.chunk_len
        # batch_size, num_lead, num_chunks, chunk_len
        x = torch.reshape(x, (bs, self.num_lead, num_chunks, self.chunk_len))
        x = x.permute(0, 2, 1, 3)

        # batch_size, num_chunks, num_lead * chunk_len
        x = torch.reshape(x, (bs, num_chunks, -1))

        feat = self.embed(x)
        return feat

class ConvEmbed(nn.Module):

    def __init__(self, num_lead: int, d_model: int) -> None:
        super(ConvEmbed, self).__init__()

        self.num_lead = num_lead

        self.conv1 = nn.Conv1d(num_lead, 128, kernel_size=14, stride=3, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.dense = nn.Linear(256, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, seqlen).
        Returns:
            feat (torch.Tensor): Tensor of size (batch_size, num_steps, d_model).
        """
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)
        feat = self.conv5(feat)
        feat = self.conv6(feat)

        feat = feat.permute(0, 2, 1)
        feat = self.dense(feat)

        return feat

class PositionalEncoding(nn.Module):
    """
    From `https://pytorch.org/tutorials/beginner/transformer_tutorial.html`
    """

    def __init__(
        self,
        d_model: int,
        dropout: float=0.1,
        max_len: int=5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        Add positional encoding.
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_steps, d_model).
        Returns:
            x (torch.Tensor): Tensor of size (batch_size, num_steps, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        embed_mode: str,
        num_lead: int = 12,
        chunk_len: int = 50,
        embedding_dim: int = 512,
        **kwargs,
    ) -> None:
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        print("Transformer with embedding dim {}".format(self.embedding_dim))

        if embed_mode == "linear":
            self.embed = LinearEmbed(num_lead, chunk_len, d_model)
        elif embed_mode == "cnn":
            self.embed = ConvEmbed(num_lead, d_model)
        else:
            raise NotImplementedError(f"Embedding model `{embed_mode}` is not implemented")

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, self.embedding_dim)

        self.name = "transformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, seqlen).
        Returns:
            feat (torch.Tensor): Tensor of size (batch_size, embedding_dim).
        """

        feat = self.embed(x)
        # feat = torch.clamp(feat, min=-1*clamp_val, max=clamp_val)
        feat = self.pos_encoder(feat)
        feat = self.transformer_encoder(feat)

        # batch_size, num_chunks, ff_dim -> batch_size, ff_dim, num_chunks
        feat = feat.permute(0, 2, 1)
        feat = nn.AdaptiveAvgPool1d(1)(feat)
        feat = feat.squeeze(-1)
        feat = self.fc(feat)
        return feat

def _transformer(
    arch: str,
    num_layers: int,
    num_heads: int,
    d_model: int,
    ff_dim: int,
    embed_mode: str,
    **kwargs: Any,
) -> TransformerModel:
    model = TransformerModel(
        num_layers, num_heads, d_model, ff_dim, embed_mode, **kwargs)
    return model

# Depth = 1 (For model checking)
def transformer_d1_h1_dim32l(**kwargs: Any) -> TransformerModel:
    num_layers = 1
    num_heads = 1
    d_model = 32
    ff_dim = 32
    embed_mode = "linear"
    return _transformer("transformer_d1_h1_dim32l", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d1_h1_dim32c(**kwargs: Any) -> TransformerModel:
    num_layers = 1
    num_heads = 1
    d_model = 32
    ff_dim = 32
    embed_mode = "cnn"
    return _transformer("transformer_d1_h1_dim32c", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

# Linear embed models.
def transformer_d2_h4_dim64l(**kwargs: Any) -> TransformerModel:
    num_layers = 2
    num_heads = 4
    d_model = 32
    ff_dim = 64
    embed_mode = "linear"
    return _transformer("transformer_d2_h4_dim32l", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d4_h4_dim64l(**kwargs: Any) -> TransformerModel:
    num_layers = 4
    num_heads = 4
    d_model = 32
    ff_dim = 64
    embed_mode = "linear"
    return _transformer("transformer_d4_h4_dim32l", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d8_h4_dim64l(**kwargs: Any) -> TransformerModel:
    num_layers = 8
    num_heads = 4
    d_model = 32
    ff_dim = 64
    embed_mode = "linear"
    return _transformer("transformer_d8_h4_dim32l", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d8_h8_dim256l(**kwargs: Any) -> TransformerModel:
    num_layers = 8
    num_heads = 8
    d_model = 2048
    ff_dim = 256
    embed_mode = "linear"
    return _transformer("transformer_d8_h8_dim256l", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

# CNN embed models.
def transformer_d8_h4_dim64c(**kwargs: Any) -> TransformerModel:
    num_layers = 8
    num_heads = 4
    d_model = 32
    ff_dim = 64
    embed_mode = "cnn"
    return _transformer("transformer_d8_h4_dim32c", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d2_h8_dim256c(**kwargs: Any) -> TransformerModel:
    num_layers = 2
    num_heads = 8
    d_model = 256
    ff_dim = 2048
    embed_mode = "cnn"
    return _transformer("transformer_d2_h8_dim256c", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d4_h8_dim256c(**kwargs: Any) -> TransformerModel:
    num_layers = 4
    num_heads = 8
    d_model = 256
    ff_dim = 2048
    embed_mode = "cnn"
    return _transformer("transformer_d4_h8_dim256c", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)

def transformer_d8_h8_dim256c(**kwargs: Any) -> TransformerModel:
    num_layers = 8
    num_heads = 8
    d_model = 256
    ff_dim = 2048
    embed_mode = "cnn"
    return _transformer("transformer_d8_h8_dim256c", num_layers, num_heads,
                        d_model, ff_dim, embed_mode, **kwargs)