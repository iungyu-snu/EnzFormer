import argparse
import torch
import torch.nn as nn
import glob
from protein_embedding import ProteinEmbedding
from model import Linear_esm
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Evaluate your model with specific data")
    parser.add_argument("model_location", type=str, help="The name of the ESM BERT model")
    parser.add_argument("fasta_path", type=str, help="only one fasta file or fasta files directory ")
    parser.add_argument("model_checkpoint", type=str, help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument("output_dim", type=int, help="Number of groups to classify")
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks used in the model")
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args =parser.parse_args()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    valid_model_locations = [
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
    ]

    if args.model_location not in valid_model_locations:
        print(f"Error: Invalid model_location '{args.model_location}'. Must be one of: {', '.join(valid_model_locations)}")
        sys.exit(1)

    model_location = args.model_location

    if model_location == "esm2_t48_15B_UR50D":
        num_layers = 47
    elif model_location == "esm2_t36_3B_UR50D":
        num_layers = 35
    elif model_location == "esm2_t33_650M_UR50D":
        num_layers = 32
    elif model_location == "esm2_t30_150M_UR50D":
        num_layers = 29
    elif model_location == "esm2_t12_35M_UR50D":
        num_layers = 11
    elif model_location == "esm2_t6_8M_UR50D":
        num_layers = 5
    else:
        print(f"Error: Unknown model location '{model_location}'.")
        sys.exit(1)

    output_dim = args.output_dim
    num_blocks = args.num_blocks

    try:
        model = Linear_esm(model_location, output_dim, num_blocks, num_layers).to(device)
    except Exception as e:
        print(f"Error: Unable to create Linear_esm model: {e}")
        sys.exit(1)

    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint '{args.model_checkpoint}' not found")
        sys.exit(1)
    
    try:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error: Unable to load model checkpoint '{args.model_checkpoint}': {e}")
        sys.exit(1)

    model.eval()

    fasta_path = args.fasta_path

    if os.path.isdir(fasta_path):
        fasta_files= [
            f
            for f in glob.glob(os.path.join(fasta_path, "*.fasta"))
            if not f.endswith("_temp.fasta")
        ]
    else:
        if not os.path.isfile(fasta_path):
            print(f"Error: Fasta File '{fasta_path}' not found")
            sys.exit(1)
        fasta_files = [fasta_path]

    if not fasta_files:
        print(f"Error: No fasta files found in '{fasta_path}'.")
        sys.exit(1)


    for fasta_file in fasta_files:
        try:
            with torch.no_grad():
                output = model(fasta_file).to(device)
                if output is not None:
                    print(f"Output for {fasta_file}: {output}")
                else:
                    print(f"Error: No output is provided from the model '{fasta_file}'")
                    sys.exit(1)
        except Exception as e:
            print(f"Error: No output from the model '{fasta_file}': {e}")



if __name__ == "__main__":
    main()

