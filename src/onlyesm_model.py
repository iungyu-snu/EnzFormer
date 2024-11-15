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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleEsm(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "15B":
            self.embed_dim = 5120
        elif self.model_name == "3B":
            self.embed_dim = 2560
        elif self.model_name == "650M":
            self.embed_dim = 1280
        elif self.model_name == "150M":
            self.embed_dim = 640
        elif self.model_name == "35M":
            self.embed_dim = 480
        elif self.model_name == "8M":
            self.embed_dim = 320
        else:
            raise ValueError("Provide an accurate esm_embedder name")

        self.output_dim = output_dim

        # layers
        self.layer_norm = nn.LayerNorm(self.embed_dim).to(device)
        self.class_fc = nn.Linear(self.embed_dim, self.output_dim).to(device)

    def forward(self, fasta_embeds):
        x = fasta_embeds
        x = self.layer_norm(x)
        if torch.isnan(x).any():
            print("NaNs detected after layer_norm")
        x = x.mean(dim=1)
        x = self.class_fc(x)  # x shape: [batch_size, output_dim]
        return x

