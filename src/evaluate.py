import argparse
import torch
import glob
from colorama import Fore, Style, init
init(autoreset=True)
from model import ECT
import os
import sys
import subprocess
import numpy as np
import torch.nn.functional as F

enzyme_dict = {
    0: "Cystathionine beta-lyase",
    1: "Cystathionine gamma-lyase",
    2: "Methionine gamma-lyase",
    3: "Cysteine desulfhydrase",
    4: "Selenocysteine lyase",
    5: "Tryptophanase",
    6: "Serine dehydratase"
}

RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with specific data")
    parser.add_argument("model_name", type=str, help="Name of the ESM BERT model")
    parser.add_argument("fasta_path", type=str, help="FASTA file or directory")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint (.pth file)")
    parser.add_argument("output_dim", type=int, help="Number of classification groups")
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks in the model")
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    model_layers = {
        "15B": 47,
        "3B": 35,
        "650M": 32,
        "150M": 29,
        "35M": 11,
        "8M": 5
    }
    num_layers = model_layers.get(args.model_name)

    # Initialize and load model
    model = ECT(
        model_name=args.model_name,
        output_dim=args.output_dim,
        num_blocks=args.num_blocks,
        n_head=4,
        dropout_rate=0.45,
    ).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.eval()

    # Get FASTA files
    fasta_files = (
        glob.glob(os.path.join(args.fasta_path, "*.fasta")) if os.path.isdir(args.fasta_path) else [args.fasta_path]
    )
    if not fasta_files:
        print(f"No FASTA files found in '{args.fasta_path}'.")
        sys.exit(1)

    # Evaluate model on each FASTA file
    for fasta_file in fasta_files:
        npy_file = os.path.join(args.fasta_path, f"{os.path.splitext(os.path.basename(fasta_file))[0]}.npy")
        dm_npy_file = os.path.join(args.fasta_path, f"{os.path.splitext(os.path.basename(fasta_file))[0]}_dm.npy")


        npy_tensor = torch.from_numpy(np.load(npy_file)).float().unsqueeze(0).to(device)
        dm_npy_tensor = torch.from_numpy(np.load(dm_npy_file)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(npy_tensor, dm_npy_tensor).to(device)
            output = F.softmax(output, dim=1)
            torch.set_printoptions(sci_mode=False, precision=6)
            max_values, max_indices = torch.max(output, dim=1)
    
            max_index = max_indices.item()
            max_value = max_values.item()
            


            name=os.path.splitext(os.path.basename(fasta_file))[0]

        if max_value < 0.8:
            print(f"{Fore.RED}{Style.BRIGHT}*** CAN'T CONFIDENTLY PREDICT ENZYME FUNCTION ***")
            print(f"{Fore.YELLOW}Predicted enzyme (low confidence) for {name}: {enzyme_dict[max_index]}")
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}=== PREDICTED ENZYME FOR {name}: {enzyme_dict[max_index]} ===")
    
        print(f"{Fore.GREEN}{Style.BRIGHT}CONFIDENCE OF AI: {max_value:.6f}")





if __name__ == "__main__":
    main()

