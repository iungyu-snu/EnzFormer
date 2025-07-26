from Bio.PDB import PDBParser
import sys


def get_chain_names(pdb_file):
    # PDBParser ?? ??
    parser = PDBParser(QUIET=True)

    # PDB ?? ??
    structure = parser.get_structure("structure", pdb_file)

    # ?? ?? ??
    chain_names = [chain.id for model in structure for chain in model]

    # ???? ???? ?? (?: ['B', 'X'] -> 'B,X')
    chain_names_str = ",".join(chain_names)

    return f'"{chain_names_str}"'


# PDB ?? ??
pdb_file_path = sys.argv[1]

# ?? ?? ??
chains = get_chain_names(pdb_file_path)
print(chains)
