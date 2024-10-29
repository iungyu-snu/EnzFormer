import torch
import esm
from esm import (
    Alphabet,
    FastaBatchedDataset,
    ProteinBertModel,
    pretrained,
    MSATransformer,
)
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return self.layer_norm(x)


class Linear_piece(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Linear_block(nn.Module):
    def __init__(self, model_location, embed_dim, length):
        super(Linear_block, self).__init__()

        self.model_location = model_location
        self.embed_dim = embed_dim
        self.length = length

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.seq_mixing = Linear_piece(self.length)
        self.channel_mixing = Linear_piece(self.embed_dim)

    def forward(self, x):

        y = self.layer_norm1(x)

        y = y.permute(1, 0)  # Assuming input shape is (batch_size, length, embed_dim)

        y = self.seq_mixing(y)

        y = y.permute(1, 0)  # Swap back

        x = x + y

        y = self.layer_norm2(x)

        y = self.channel_mixing(y)

        x = x + y

        return x
