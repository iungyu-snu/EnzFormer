import sys
import numpy as np
from af2.af.prep import prep_pdb
from af2.af.model import model

def main(pdb_file,chain_id):
  pdb = prep_pdb(pdb_file, chain=chain_id, for_alphafold=False)
  x_beta, _ = model.modules.pseudo_beta_fn(aatype=pdb['batch']['aatype'],
                                             all_atom_positions=pdb['batch']["all_atom_positions"],
                                             all_atom_mask=pdb['batch']["all_atom_mask"])
  dm = np.sqrt(np.square(x_beta[:,None] - x_beta[None,:]).sum(-1))

  return dm


