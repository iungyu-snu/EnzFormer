import argparse
import torch
import glob
from colorama import Fore, Style, init
init(autoreset=True)
from model_250416 import ECT
import os
import sys
import subprocess
import numpy as np
import torch.nn.functional as F
import re # 추가: 정규 표현식 모듈
from captum.attr import IntegratedGradients

# PSSM 알파벳 순서 (mutation_pssm.py와 일치해야 함)
PSSM_ALPHABET = "-ABCDEFGHIKLMNPQRSTVWXYZUO*J"

# BLOSUM62 Matrix
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4}
}

# 중요: 이 딕셔너리는 Score 계산에 사용되는 인덱스와 일치해야 합니다.
# GOOD는 인덱스 0, BAD는 인덱스 1에 해당합니다.
enzyme_dict = {
    0: "GOOD",
    1: "BAD",
    2: "No enzyme"
}

# Score 계산을 위한 인덱스 정의
GOOD_INDEX = 0
BAD_INDEX = 1
NO_ENZYME_INDEX = 2 # 3번째 클래스 인덱스 추가

EPSILON = 1e-9 # 분모가 0이 되는 것을 방지하기 위함

RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
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
    parser = argparse.ArgumentParser(description="Evaluate model ensemble with specific data")
    parser.add_argument("model_name", type=str, help="Name of the ESM BERT model")
    parser.add_argument("fasta_path", type=str, help="FASTA file or directory containing FASTA and corresponding .npy files")
    parser.add_argument(
        "model_checkpoints", 
        type=str, 
        nargs='+', # Accept one or more checkpoint paths
        help="Path(s) to model checkpoint (.pth) files for the ensemble"
    )
    parser.add_argument("output_dim", type=int, help="Number of classification groups")
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks in the model")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads (default: 12)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate used during training (default: 0.3)")
    parser.add_argument("--reference_pssm_npy", type=str, required=True, help="Path to the reference PSSM .npy file (e.g., MccB_pssm.npy)")
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    models = []
    for i, ckpt_path in enumerate(args.model_checkpoints):
        try:
            model = ECT(
                model_name=args.model_name,
                output_dim=args.output_dim,
                num_blocks=args.num_blocks,
                n_head=args.n_head,
                dropout_rate=args.dropout_rate,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            models.append(model)
        except FileNotFoundError:
            print(f"{RED}Error: Checkpoint file not found: {ckpt_path}{RESET}")
            sys.exit(1)
        except Exception as e:
            print(f"{RED}Error loading model {ckpt_path}: {e}{RESET}")
            sys.exit(1)
    
    # Load reference PSSM matrix once
    reference_pssm_matrix = None
    if args.reference_pssm_npy:
        try:
            reference_pssm_matrix = np.load(args.reference_pssm_npy)
            if not isinstance(reference_pssm_matrix, np.ndarray) or reference_pssm_matrix.ndim != 2:
                print(f"{RED}Error: Reference PSSM data in '{args.reference_pssm_npy}' is not a 2D NumPy array.{RESET}")
                reference_pssm_matrix = None # Do not use if malformed
            elif reference_pssm_matrix.shape[1] != len(PSSM_ALPHABET):
                 print(f"{RED}Warning: Reference PSSM alphabet size ({reference_pssm_matrix.shape[1]}) "
                       f"does not match PSSM_ALPHABET size ({len(PSSM_ALPHABET)}). Delta PSSM may be incorrect.{RESET}")

        except FileNotFoundError:
            print(f"{RED}Error: Reference PSSM file not found: '{args.reference_pssm_npy}'. Delta PSSM scores will be N/A.{RESET}")
        except Exception as e:
            print(f"{RED}Error loading reference PSSM .npy file '{args.reference_pssm_npy}': {e}. Delta PSSM scores will be N/A.{RESET}")
    else:
        print(f"{RED}Warning: No reference PSSM file provided. Delta PSSM scores will be N/A.{RESET}")

    if os.path.isdir(args.fasta_path):
        fasta_files = glob.glob(os.path.join(args.fasta_path, "*.fasta"))
    elif os.path.isfile(args.fasta_path) and args.fasta_path.endswith(".fasta"):
        fasta_files = [args.fasta_path]
    else:
        print(f"{RED}Error: fasta_path must be a directory containing .fasta files or a single .fasta file.{RESET}")
        sys.exit(1)

    if not fasta_files:
        print(f"No FASTA files found in '{args.fasta_path}'.")
        sys.exit(1)

    print("Point_mutation_name | label_from_AI | Confidence | probability for GOOD | probability for BAD | probability for No enzyme | Delta_PSSM | Delta_BLOSUM62 | Custom_Score | Top_5_Residues_IG | Top_5_Residues_Attn")

    for fasta_file in fasta_files:
        base_name = os.path.splitext(os.path.basename(fasta_file))[0]
        npy_file = os.path.join(os.path.dirname(fasta_file), f"{base_name}.npy")

        if not os.path.exists(npy_file):
            print(f"Warning: Corresponding .npy file not found for {fasta_file}. Skipping.", file=sys.stderr)
            continue

        try:
            npy_data = np.load(npy_file)
            npy_tensor = torch.from_numpy(npy_data).float().unsqueeze(0).to(device)
            npy_tensor.requires_grad = True # IG 계산을 위해 그래디언트 추적 활성화
        except Exception as e:
            print(f"Error loading .npy file {npy_file}: {e}", file=sys.stderr)
            continue

        all_probs_list = []
        with torch.no_grad():
            for model in models:
                output = model(npy_tensor)
                probs = F.softmax(output, dim=1)
                all_probs_list.append(probs)

        avg_probs = torch.stack(all_probs_list).mean(dim=0)
        torch.set_printoptions(sci_mode=False, precision=6)
        max_values, max_indices = torch.max(avg_probs, dim=1)

        final_pred_index = max_indices.item()
        final_confidence = max_values.item()
        name = base_name
        
        # 3개 클래스에 대한 확률을 안전하게 추출
        prob_for_good = avg_probs[0, GOOD_INDEX].item() if avg_probs.shape[1] > GOOD_INDEX else 0.0
        prob_for_bad = avg_probs[0, BAD_INDEX].item() if avg_probs.shape[1] > BAD_INDEX else 0.0
        prob_for_no_enzyme = avg_probs[0, NO_ENZYME_INDEX].item() if avg_probs.shape[1] > NO_ENZYME_INDEX else 0.0
        # 각 단계에서 기울기를 구하고, 그 기울기를 평균 내면 대표 기울기. 그 대표 기우릭와 전체 변화량을 곱하면 IG 값        
        # Integrated Gradients 계산
        all_attributions = []
        baseline = torch.zeros_like(npy_tensor)
        for model in models:
            ig = IntegratedGradients(model)
            attributions = ig.attribute(npy_tensor, baselines=baseline, target=final_pred_index, n_steps=50)
            all_attributions.append(attributions)
        
        avg_attributions = torch.stack(all_attributions).mean(dim=0)
        
        attributions_np = avg_attributions.squeeze(0).cpu().detach().numpy()
        attr_magnitudes = np.linalg.norm(attributions_np, axis=1)
        
        # 중요도 상위 5개 잔기 추출
        top_5_indices = np.argsort(attr_magnitudes)[-5:][::-1]
        top_5_scores = attr_magnitudes[top_5_indices]
        
        top_5_ig_residues_str = ", ".join([f"{idx+1}({score:.4f})" for idx, score in zip(top_5_indices, top_5_scores)])

        # Attention Weight Analysis
        all_attentions_list = []
        with torch.no_grad():
            for model in models:
                # The first return value is output, which we don't need here
                _, attentions = model(npy_tensor, return_attentions=True)
                all_attentions_list.append(attentions)

        # Average attentions across the ensemble
        avg_attentions = torch.stack(all_attentions_list).mean(dim=0)
        
        # Average across layers and heads. Shape: (num_blocks, B, H, L, L) -> (B, L, L)
        res_res_attention = avg_attentions.mean(dim=(0, 2)).squeeze(0) # now (L, L)

        # Get a score for each residue by summing the attention it receives from all others
        per_residue_attention_scores = res_res_attention.sum(dim=0) # Sum over rows (source residues)

        # Get top 5 residues based on attention
        top_5_attn_indices = torch.argsort(per_residue_attention_scores, descending=True)[:5]
        top_5_attn_scores = per_residue_attention_scores[top_5_attn_indices]

        top_5_attn_residues_str = ", ".join([f"{idx+1}({score:.4f})" for idx, score in zip(top_5_attn_indices.cpu().numpy(), top_5_attn_scores.cpu().numpy())])

        delta_score_val = np.nan # Use np.nan for numerical calculations if N/A
        delta_score_str = "N/A"
        delta_blossum_score_str = "N/A" # BLOSUM 점수 변수 추가

        original_aa, position_0_indexed, new_aa = parse_mutation_string(base_name)
        
        if reference_pssm_matrix is not None:
            if original_aa is not None:
                original_aa_idx = get_aa_index(original_aa, PSSM_ALPHABET)
                new_aa_idx = get_aa_index(new_aa, PSSM_ALPHABET)
                
                if original_aa_idx is not None and new_aa_idx is not None:
                    seq_len_pssm, alphabet_size_pssm = reference_pssm_matrix.shape
                    if 0 <= position_0_indexed < seq_len_pssm:
                        try:
                            score_original = reference_pssm_matrix[position_0_indexed, original_aa_idx]
                            score_mutated = reference_pssm_matrix[position_0_indexed, new_aa_idx]
                            delta_score_val = score_mutated - score_original
                            delta_score_str = f"{delta_score_val}"
                        except IndexError: pass
        
        # Delta BLOSUM62 점수 계산
        if original_aa and new_aa:
            # BLOSUM62 딕셔너리에서 점수 조회 (존재하지 않는 키는 .get()으로 안전하게 처리)
            score = BLOSUM62.get(original_aa, {}).get(new_aa)
            if score is not None:
                delta_blossum_score_str = f"{score}"
        
        # Calculate Custom Score
        custom_score_str = "N/A"
        if not np.isnan(delta_score_val):
            log_term_numerator = prob_for_good
            # 3중 분류이므로 분모를 P(BAD) + P(No enzyme)로 일반화합니다.
            log_term_denominator = prob_for_bad + prob_for_no_enzyme + EPSILON
            
            # 확률은 음수일 수 없으며 분모는 양수여야 합니다.
            if log_term_denominator <= EPSILON or log_term_numerator < 0:
                log_value = np.nan # 잘못된 값
            elif log_term_numerator == 0:
                log_value = -np.inf # log(0)은 -inf
            else:
                try:
                    log_value = np.log(log_term_numerator / log_term_denominator)
                except (ValueError, ZeroDivisionError): 
                    log_value = np.nan # 계산 오류
            
            if np.isinf(log_value) or np.isnan(log_value):
                # log_value에 문제가 있는 경우 점수가 의미 없거나 계산할 수 없습니다.
                custom_score_str = "N/A (log error)" 
            else:
                custom_score = 0.8 * log_value + 0.2 * delta_score_val
                custom_score_str = f"{custom_score:.6f}"
        else: # delta_score_val is np.nan
            custom_score_str = "N/A (no delta PSSM)"

        print(f"{name} | {enzyme_dict.get(final_pred_index, 'Unknown_Label')} | {final_confidence:.6f} | {prob_for_good:.6f} | {prob_for_bad:.6f} | {prob_for_no_enzyme:.6f} | {delta_score_str} | {delta_blossum_score_str} | {custom_score_str} | {top_5_ig_residues_str} | {top_5_attn_residues_str}")

if __name__ == "__main__":
    main()

