import argparse
import torch
import glob
from colorama import Fore, Style, init
init(autoreset=True)
from no_dm_model import ECT
import os
import sys
import subprocess
import numpy as np
import torch.nn.functional as F

enzyme_dict = {
    0: "4.4.1.1",
    1: "4.4.1.11",
    2: "4.4.1.13",
    3: "4.4.1.14",
    4: "4.4.1.15",
    5: "4.4.1.16",
    6: "4.4.1.17",
    7: "4.4.1.19",
    8: "4.4.1.2",
    9: "4.4.1.20",
    10: "4.4.1.21",
    11: "4.4.1.22",
    12: "4.4.1.23",
    13: "4.4.1.24",
    14: "4.4.1.25",
    15: "4.4.1.3",
    16: "4.4.1.5",
    17: "4.4.1.9",
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

    model = ECT(
        model_name=args.model_name,
        output_dim=args.output_dim,
        num_blocks=args.num_blocks,
        n_head=12,
        dropout_rate=0.3,
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


        npy_tensor = torch.from_numpy(np.load(npy_file)).float().unsqueeze(0).to(device)
        npy_tensor.requires_grad = True
        output = model(npy_tensor).to(device)
        output = F.softmax(output, dim=1)
        torch.set_printoptions(sci_mode=False, precision=6)
        max_values, max_indices = torch.max(output, dim=1)
    
        max_index = max_indices.item()
        max_value = max_values.item()
            


        name=os.path.splitext(os.path.basename(fasta_file))[0]
        model.zero_grad()
        # Only choose first batch's max_index 
        output[0, max_index].backward()
        
        gradients = npy_tensor.grad.data.cpu().numpy()[0]  # Remove batch dimension

        # calculate the embedding vectors to one scalar.
        grad_magnitudes = np.linalg.norm(gradients, axis=1)
        normalized_grad_magnitudes = grad_magnitudes / np.sum(grad_magnitudes)
        top_5_indices = np.argsort(normalized_grad_magnitudes)[-5:][::-1] 
        top_5_probs = normalized_grad_magnitudes[top_5_indices]  

        for i, (index, prob) in enumerate(zip(top_5_indices, top_5_probs), start=1):
            print(f"Top {i}: Index = {index + 1}, Probability = {prob:.6f}")


        if max_value < 0.8:

            print(f"{Fore.YELLOW}Predicted enzyme (low confidence) for {name}: {enzyme_dict[max_index]}: CONFIDENCE OF AI: {max_value:.6f}")
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}PREDICTED ENZYME FOR {name}: {enzyme_dict[max_index]}: CONFIDENCE OF AI: {max_value:.6f}")






if __name__ == "__main__":
    main()

