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
import math
from torch import einsum
from torch.nn.functional import softmax


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * r_ff)
        self.dropout1 = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model * r_ff, d_model)
        self.dropout2 = nn.Dropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu_(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_query, d_key, n_head, d_hidden, p_drop=0.1):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden

        self.to_q = nn.Linear(d_query, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head * d_hidden, bias=False)
        self.to_out = nn.Linear(
            n_head * d_hidden, d_query
        )  # Assuming output dim is same as input dim

        self.scaling = 1 / math.sqrt(d_hidden)

        self.attn_dropout = nn.Dropout(p_drop)
        self.out_dropout = nn.Dropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value, dm_embeds):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]

        query_proj = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key_proj = self.to_k(key).reshape(B, K, self.h, self.dim)
        value_proj = self.to_v(value).reshape(B, K, self.h, self.dim)

        query_proj = query_proj * self.scaling
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", query_proj, key_proj)

        # =====
        # Adding pair bias to attention scores
        #        dm_embeds = dm_embeds.unsqueeze(1).repeat(1, self.h, 1, 1)
        #        attn_scores = attn_scores + dm_embeds

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value_proj)
        attn_output = attn_output.reshape(B, Q, self.h * self.dim)
        attn_output = self.to_out(attn_output)
        attn_output = self.out_dropout(attn_output)

        return attn_output


class TransBlock(nn.Module):
    def __init__(self, model_name, n_head, r_ff=4, p_drop=0):
        super().__init__()
        if model_name == "15B":
            self.embed_dim = 5120
        elif model_name == "3B":
            self.embed_dim = 2560
        elif model_name == "650M":
            self.embed_dim = 1280
        elif model_name == "150M":
            self.embed_dim = 640
        elif model_name == "35M":
            self.embed_dim = 480
        elif model_name == "8M":
            self.embed_dim = 320
        else:
            raise ValueError("Provide an accurate esm_embedder name")

        self.n_head = n_head
        self.d_hidden = self.embed_dim // self.n_head
        self.r_ff = r_ff

        # Layers
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

        self.attention = Attention(
            self.embed_dim, self.embed_dim, self.n_head, self.d_hidden, p_drop=p_drop
        )
        self.feedforward = FeedForwardLayer(self.embed_dim, self.r_ff, p_drop=p_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Self-attention block with residual connection
        residual = x
        x_norm = self.layer_norm1(x)
        attn_output = self.attention(x_norm, x_norm, x_norm, y)
        x = residual + self.dropout1(attn_output)

        # Feed-forward block with residual connection
        residual = x
        x_norm = self.layer_norm2(x)
        ff_output = self.feedforward(x_norm)
        x = residual + self.dropout2(ff_output)

        return x
