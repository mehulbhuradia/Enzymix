import os
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm


def average_bfactor(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', pdb_file)
    # Initialize a count for residues
    num_residues = 0

    # Iterate through each model in the structure
    for model in structure:
        # Iterate through each chain in the model
        for chain in model:
            # Add the number of residues in this chain to the total count
            num_residues += len(list(chain.get_residues()))
    
    bfactor_values = [atom.get_bfactor() for atom in structure.get_atoms() if atom.get_bfactor() is not None]
    if bfactor_values:
        return np.mean(bfactor_values),num_residues
    else:
        return None,None


def clean_pdb_folder(folder_path):    
    results = []
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdb')]
    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        for filename in file_list:
            pdb_file = os.path.join(folder_path, filename)
            average,siz = average_bfactor(pdb_file)
            pbar.update(1)
            if siz >=50 and siz <= 100 and average < 70:
                results.append(average)
            else:
                os.remove(pdb_file)
                print(f"Removed {filename}")
    print(f"Number of files removed: {len(file_list)-len(results)}")
    print(f"Number of files kept: {len(results)}")
    print(f"Average Bfactor of kept files: {np.mean(results)}")


clean_pdb_folder("./swiss_prot_pdbs/")
