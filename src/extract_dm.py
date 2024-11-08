import numpy as np
from af2.af.prep import prep_pdb
from af2.af.model import model
import sys
import os
import tensorflow as tf
import torch

def configure_tf_and_torch():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    torch.cuda.empty_cache()


def calculate_distance_matrix(pdb_file, chain_id):
    configure_tf_and_torch()

    pdb = prep_pdb(pdb_file, chain=chain_id, for_alphafold=False)
    x_beta, _ = model.modules.pseudo_beta_fn(
        aatype=pdb["batch"]["aatype"],
        all_atom_positions=pdb["batch"]["all_atom_positions"],
        all_atom_mask=pdb["batch"]["all_atom_mask"],
    )
    return np.sqrt(np.square(x_beta[:, None] - x_beta[None, :]).sum(-1))


def main():
    pdb_file = sys.argv[1]
    chain_id = "A"
    dm = calculate_distance_matrix(pdb_file, chain_id)
    output_file = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_dm.npy"
    np.save(output_file, dm)

if __name__ == "__main__":
    main()

