import os
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

def average_bfactor(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', pdb_file)
    
    bfactor_values = [atom.get_bfactor() for atom in structure.get_atoms() if atom.get_bfactor() is not None]
    
    if bfactor_values:
        return np.mean(bfactor_values)
    else:
        return None

def process_pdb_folder(folder_path):
    results = {}
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdb')]
    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        for filename in file_list:
            pdb_file = os.path.join(folder_path, filename)
            average = average_bfactor(pdb_file)
            if average is not None:
                results[filename] = average
            else:
                print(f"No B-factor values found in {filename}")
            pbar.update(1)
    
    # Calculate overall average
    overall_average = np.mean(list(results.values()))

    # print("Average B-factor for each file:")
    # for filename, average in results.items():
    #     print(f"{filename}: {average:.2f}")

    print(f"\nOverall average B-factor: {overall_average:.2f}")

def calculate_and_print_average(folder_path):
    process_pdb_folder(folder_path)

folder_path = './OmegaOut/out_og/'
calculate_and_print_average(folder_path)
