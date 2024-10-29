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
from transformer import FeedForwardLayer, Attention, TransBlock
from protein_embedding import ProteinEmbedding
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ECT(nn.Module):
    def __init__(
        self, model_name, output_dim, num_blocks, dropout_rate=0.1
    ):
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



        # =======
        # parameters
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # ========
        # layers
        self.layer_norm = nn.LayerNorm(self.embed_dim).to(
            device
        )  
        self.class_fc = nn.Linear(self.embed_dim, self.output_dim).to(
            device
        ) 
        # ========
        # blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransBlock(self.model_name, self.embed_dim, r_ff=4, p_drop=dropout_rate, n_head=20).to(device)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, fasta_embeds):
        batch_size = len(fasta_embeds)
        prot_embeds = []
        for fasta_embed in fasta_embeds:
            fasta_embed.to(device)
            prot_embeds.append(fasta_embed)
       
        x = torch.stack(prot_embeds, dim=0)  
        if torch.isnan(x).any():
            print("NaNs detected after pad_and_stack")
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        if torch.isnan(x).any():
            print("NaNs detected after transformer")            
        x = self.layer_norm(x)
        if torch.isnan(x).any():
            print("NaNs detected after layer_norm")
        x = x.mean(dim=1)
        x = self.class_fc(x)  # x shape: [batch_size, output_dim]
        x = F.softmax(x, dim=-1)
        return x


#    def pad_and_stack(self, prot_embeds):
#        max_seq_length = max([embed.shape[0] for embed in prot_embeds])
#        batch_embeds = torch.zeros((len(prot_embeds), max_seq_length, self.embed_dim)).to(device)

#        for i, embed in enumerate(prot_embeds):
#            seq_length = embed.shape[0]
#            if seq_length < max_seq_length:
#                pad_length = max_seq_length - seq_length
#                padding = torch.zeros((pad_length, self.embed_dim)).to(device)
#                embed = torch.cat((embed, padding), dim=0)
#            batch_embeds[i] = embed
        
#        return batch_embeds
        