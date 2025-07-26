#!/usr/bin/env python3
import sys
import os
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def read_fasta_sequence(fasta_file):
    """Reads the first sequence from a FASTA file."""
    sequence = ""
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                if sequence:  # Stop after the first sequence if multiple are present
                    break
                continue
            sequence += line
    if not sequence:
        raise ValueError(f"No sequence found in {fasta_file}")
    return sequence

if len(sys.argv) < 2:
    print("Usage: python script_name.py <fasta_file>")
    sys.exit(1)

fasta_file = sys.argv[1]

try:
    # Read sequence from FASTA file provided as a command-line argument
    protein_sequence = read_fasta_sequence(fasta_file)
    protein = ESMProtein(sequence=protein_sequence)

    # --- ESM Model Loading and Inference ---
    # Consider making the device configurable (e.g., via another argument or env var)
    device = "cuda" # or "cpu"
    client = ESMC.from_pretrained("esmc_600m").to(device)

    protein_tensor = client.encode(protein)

    logits_output = client.logits(
       protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    # --- End ESM ---

    # Process embeddings: remove the first dimension (batch dimension)
    # Move tensor to CPU and convert to NumPy array for saving
    embeddings = logits_output.embeddings.squeeze(0).cpu().numpy()
    print(f"embedding shape: {embeddings.shape}")


    # Determine output filename
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    output_filename = f"{base_name}.npy"

    # Save embeddings to a .npy file
    np.save(output_filename, embeddings)


except FileNotFoundError:
    print(f"Error: Input FASTA file not found at {fasta_file}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)


