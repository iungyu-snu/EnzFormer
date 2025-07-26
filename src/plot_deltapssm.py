import argparse
import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PSSM 알파벳 순서 (mutation_pssm.py와 일치해야 함)
PSSM_ALPHABET = "-ABCDEFGHIKLMNPQRSTVWXYZUO*J"

# 색상 코드 (오류 메시지용)
RED = "\033[91m"
RESET = "\033[0m"

def parse_mutation_string(mutation_str):
    """
    Parses a mutation string like "Y26S" into original AA, position (0-indexed), and new AA.
    Returns (original_aa, position_0_indexed, new_aa) or None if parsing fails.
    """
    match = re.fullmatch(r"([A-Z*])(\d+)([A-Z*])", mutation_str.upper())
    if not match:
        return None, None, None
    
    original_aa = match.group(1)
    position_1_indexed = int(match.group(2))
    new_aa = match.group(3)
    
    if position_1_indexed == 0:
        return None, None, None
        
    return original_aa, position_1_indexed - 1, new_aa

def get_aa_index(aa_char, alphabet):
    """
    Finds the index of an amino acid character in the PSSM alphabet.
    Returns the index or None if not found.
    """
    try:
        return alphabet.index(aa_char)
    except ValueError:
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate Delta PSSM scores and plot their distribution.")
    parser.add_argument("--save_plot", type=str, help="Path to save the plot image (e.g., 'distribution.png'). If not provided, the plot is shown on screen.")
    args = parser.parse_args()
    
    # --- Configuration: Hardcoded paths ---
    fasta_path = "/nashome/uglee/EnzFormer/test_fasta/first_pointmutation_600M"
    reference_pssm_npy = "/nashome/uglee/EnzFormer/MccB_pssm2.npy"
    # --- End of Configuration ---

    # Load reference PSSM matrix once
    try:
        reference_pssm_matrix = np.load(reference_pssm_npy)
        if not isinstance(reference_pssm_matrix, np.ndarray) or reference_pssm_matrix.ndim != 2:
            print(f"{RED}Error: Reference PSSM data in '{reference_pssm_npy}' is not a 2D NumPy array.{RESET}", file=sys.stderr)
            sys.exit(1)
        if reference_pssm_matrix.shape[1] != len(PSSM_ALPHABET):
            print(f"{RED}Warning: Reference PSSM alphabet size ({reference_pssm_matrix.shape[1]}) "
                  f"does not match PSSM_ALPHABET size ({len(PSSM_ALPHABET)}). Delta PSSM may be incorrect.{RESET}", file=sys.stderr)

    except FileNotFoundError:
        print(f"{RED}Error: Reference PSSM file not found: '{reference_pssm_npy}'.{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Error loading reference PSSM .npy file '{reference_pssm_npy}': {e}.{RESET}", file=sys.stderr)
        sys.exit(1)

    # Find FASTA files
    if os.path.isdir(fasta_path):
        fasta_files = glob.glob(os.path.join(fasta_path, "*.fasta"))
    elif os.path.isfile(fasta_path) and fasta_path.endswith(".fasta"):
        fasta_files = [fasta_path]
    else:
        print(f"{RED}Error: fasta_path must be a directory containing .fasta files or a single .fasta file.{RESET}", file=sys.stderr)
        sys.exit(1)

    if not fasta_files:
        # No error message if no files are found, to keep output clean
        sys.exit(0)

    delta_scores = []
    # Process each file
    for fasta_file in fasta_files:
        base_name = os.path.splitext(os.path.basename(fasta_file))[0]
        
        original_aa, position_0_indexed, new_aa = parse_mutation_string(base_name)
        
        if original_aa is None:
            # Silently skip files with invalid names
            continue
            
        original_aa_idx = get_aa_index(original_aa, PSSM_ALPHABET)
        new_aa_idx = get_aa_index(new_aa, PSSM_ALPHABET)
        
        if original_aa_idx is not None and new_aa_idx is not None:
            seq_len_pssm, _ = reference_pssm_matrix.shape
            if 0 <= position_0_indexed < seq_len_pssm:
                try:
                    score_original = reference_pssm_matrix[position_0_indexed, original_aa_idx]
                    score_mutated = reference_pssm_matrix[position_0_indexed, new_aa_idx]
                    delta_score_val = score_mutated - score_original
                    delta_scores.append(delta_score_val)
                except IndexError:
                    # This position might be out of bounds for the PSSM matrix
                    pass

    if not delta_scores:
        print("No valid Delta PSSM scores were calculated to plot.", file=sys.stderr)
        sys.exit(0)

    # --- Plotting ---
    average_score = np.mean(delta_scores)
    print(f"Total valid Delta PSSM scores analyzed: {len(delta_scores)}")
    print(f"Average Delta PSSM score: {average_score:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    sns.histplot(delta_scores, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(average_score, color='red', linestyle='--', linewidth=2, label=f'Average: {average_score:.4f}')
    plt.title('Delta PSSM Score Distribution', fontsize=16)
    plt.xlabel('Delta PSSM', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()

    if args.save_plot:
        try:
            plt.savefig(args.save_plot)
            print(f"Plot saved to '{args.save_plot}'")
        except Exception as e:
            print(f"Error: Could not save plot to '{args.save_plot}': {e}", file=sys.stderr)
    else:
        plt.show()

if __name__ == "__main__":
    main() 