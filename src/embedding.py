import argparse
import pathlib
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
import glob
import os
import numpy as np
import sys


class ProteinEmbedding:
    def __init__(
        self,
        model_location,
        fasta_file,
        num_layers,
        include: list = ["mean", "per_tok", "bos", "contacts"],
        toks_per_batch=4096,
        truncation_seq_length=1022,
        repr_layers=None,
        nogpu=False,
    ):
        self.model_location = model_location
        self.fasta_file = fasta_file
        self.include = include
        self.toks_per_batch = toks_per_batch
        self.truncation_seq_length = truncation_seq_length
        self.repr_layers = repr_layers
        self.nogpu = nogpu
        self.num_layers = num_layers

        self._init_submodules()

    def _init_submodules(self):
        # Load the pretrained model and alphabet
        self.model, self.alphabet = pretrained.load_model_and_alphabet(
            self.model_location
        )
        self.model.eval()
        if torch.cuda.is_available() and not self.nogpu:
            self.model = self.model.cuda()
        self.dataset = FastaBatchedDataset.from_file(self.fasta_file)
        self.batches = self.dataset.get_batch_indices(
            self.toks_per_batch, extra_toks_per_seq=1
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.alphabet.get_batch_converter(self.truncation_seq_length),
            batch_sampler=self.batches,
        )
        self.all_results = None

    def forward(self):
        return_contacts = "contacts" in self.include
        repr_layers = list(range(self.model.num_layers + 1))
        all_results = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(self.data_loader):
                if torch.cuda.is_available() and not self.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.model(
                    toks, repr_layers=repr_layers, return_contacts=return_contacts
                )

                logits = out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }
                if return_contacts:
                    contacts = out["contacts"].to(device="cpu")

                for i, label in enumerate(labels):
                    result = {"label": label}
                    truncate_len = min(self.truncation_seq_length, len(strs[i]))
                    if "per_tok" in self.include:
                        result["representations"] = {
                            layer: t[i, 1 : truncate_len + 1].clone()
                            for layer, t in representations.items()
                        }
                    if "mean" in self.include:
                        result["mean_representations"] = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }
                    if "bos" in self.include:
                        result["bos_representations"] = {
                            layer: t[i, 0].clone()
                            for layer, t in representations.items()
                        }
                    if return_contacts:
                        result["contacts"] = contacts[
                            i, :truncate_len, :truncate_len
                        ].clone()

                    all_results.append(result)

        self.all_results = all_results

        results = []

        for result in self.all_results:
            rep = result.get("representations")
            result_f = rep[self.num_layers]
            results.append(result_f)
        final_result = torch.cat(results, dim=0)
        return final_result


# Define `name` at the top level so it can be imported
def get_name_from_fasta(fasta_file):
    return os.path.splitext(os.path.basename(fasta_file))[0]

def main():
    # Check if the FASTA file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python embedding.py <fasta_file>")
        sys.exit(1)

    model_location = "esm2_t6_8M_UR50D"
    num_layers = 5
    fasta_file = sys.argv[1]
    
    # Generate the protein embedding
    prot_embed = ProteinEmbedding(model_location, fasta_file, num_layers).forward()

    # Extract the filename without the extension and save as .npy
    name = get_name_from_fasta(fasta_file)
    output_path = f"{name}.npy"
    np.save(output_path, prot_embed)

    print(f"Protein embedding saved to {output_path}")


# Make sure main runs only when script is executed directly
if __name__ == "__main__":
    main()

