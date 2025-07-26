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
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence
import logging  # Import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Centralized mapping for model name to embedding dimension
MODEL_EMBED_DIMS = {
    "15B": 5120,
    "3B": 2560,
    "600M": 1152,
    "650M": 1280,
    "150M": 640,
    "35M": 480,
    "8M": 320,
}

def get_embed_dim(model_name: str) -> int:
    """Gets the embedding dimension for a given ESM model name."""
    try:
        return MODEL_EMBED_DIMS[model_name]
    except KeyError:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models: {list(MODEL_EMBED_DIMS.keys())}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, query, key, value):
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

        return attn_output, attn_weights


class TransBlock(nn.Module):
    """
    A single Transformer block consisting of self-attention and feed-forward layers.
    """
    def __init__(self, embed_dim, n_head, r_ff=4, p_drop=0.1):
        """
        Initializes the TransBlock.

        Args:
            embed_dim (int): The embedding dimension of the input and output.
            n_head (int): The number of attention heads.
            r_ff (int): The expansion ratio for the feed-forward layer. Defaults to 4.
            p_drop (float): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        if self.embed_dim % self.n_head != 0:
             raise ValueError(f"Embedding dimension ({self.embed_dim}) must be divisible by the number of heads ({self.n_head})")
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

    def forward(self, x: torch.Tensor, return_attentions: bool = False):
        """Forward pass through the Transformer block."""
        # Self-attention block with residual connection
        residual = x
        x_norm = self.layer_norm1(x)
        attn_output, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = residual + self.dropout1(attn_output)

        # Feed-forward block with residual connection
        residual = x
        x_norm = self.layer_norm2(x)
        ff_output = self.feedforward(x_norm)
        x = residual + self.dropout2(ff_output)

        if return_attentions:
            return x, attn_weights
        return x


class ECT(nn.Module):
    """
    EnzFormer Classification Transformer (ECT).
    Applies Transformer blocks to sequence embeddings for classification.
    """
    def __init__(self, model_name, output_dim, num_blocks, n_head, r_ff=4, dropout_rate=0.1):
        """
        Initializes the ECT model.

        Args:
            model_name (str): The name of the base ESM model (e.g., "650M", "3B").
            output_dim (int): The dimension of the final output classification layer.
            num_blocks (int): The number of Transformer blocks to stack.
            n_head (int): The number of attention heads in each Transformer block.
            r_ff (int): The expansion ratio for the feed-forward layers. Defaults to 4.
            dropout_rate (float): The dropout probability used throughout the model. Defaults to 0.1.
        """
        super().__init__()
        self.model_name = model_name
        self.embed_dim = get_embed_dim(model_name) # Use helper function

        # =======
        # parameters
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.n_head = n_head
        self.r_ff = r_ff # Make r_ff configurable
        # ========
        # layers
        self.layer_norm = nn.LayerNorm(self.embed_dim) # Remove .to(device)
        self.class_fc = nn.Linear(self.embed_dim, self.output_dim) # Remove .to(device)
        # ========
        # blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransBlock(
                    embed_dim=self.embed_dim, # Pass embed_dim directly
                    n_head=self.n_head,
                    r_ff=self.r_ff, # Use configurable r_ff
                    p_drop=dropout_rate
                ) # Remove .to(device)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, fasta_embeds: torch.Tensor, return_attentions: bool = False):
        """
        Forward pass through the ECT model.

        Args:
            fasta_embeds (torch.Tensor): Input embeddings of shape [batch_size, seq_len, embed_dim].
            return_attentions (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output classification scores of shape [batch_size, output_dim].
            or
            (torch.Tensor, torch.Tensor): Output scores and attention weights if return_attentions is True.
        """
        x = fasta_embeds
        attentions = []
        # Optional NaN check using logging (remove if not needed)
        # if torch.isnan(x).any():
        #     logging.warning("NaNs detected in input embeddings.")

        for transformer_block in self.transformer_blocks:
            if return_attentions:
                x, attn_weights = transformer_block(x, return_attentions=True)
                attentions.append(attn_weights)
            else:
                x = transformer_block(x)

        # Optional NaN check using logging
        # if torch.isnan(x).any():
        #     logging.warning("NaNs detected after transformer blocks.")

        x = self.layer_norm(x)

        # Optional NaN check using logging
        # if torch.isnan(x).any():
        #     logging.warning("NaNs detected after layer normalization.")

        # Mean pooling across sequence length dimension
        x_pooled = x.mean(dim=1)
        x_out = self.class_fc(x_pooled)
        
        if return_attentions:
            # Stack attentions: (num_blocks, batch_size, num_heads, seq_len, seq_len)
            return x_out, torch.stack(attentions)
        return x_out
