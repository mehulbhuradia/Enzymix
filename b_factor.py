import os
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default='generated/fasta/omega')
args = parser.parse_args()


def average_bfactor(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', pdb_file)
    bfactor_values = [atom.get_bfactor() for atom in structure.get_atoms() if ((atom.get_bfactor() is not None)  and (atom.get_name() == 'CA'))]
    if bfactor_values:
        return np.mean(bfactor_values)
    else:
        return None

def process_pdb_folder(folder_path):
    out_file = folder_path.replace('/omega', '/generated.json')
    with open(out_file, 'r') as f:
        output = json.load(f)
    results = []
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdb')]
    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        for filename in file_list:
            pdb_file = os.path.join(folder_path, filename)
            average = average_bfactor(pdb_file)
            pbar.update(1)
            output[filename.split(".")[0]]['average_bfactor'] = average
            results.append(average)
    
    # Calculate overall average
    overall_average = np.mean(results)

    print(f"\nOverall average B-factor: {overall_average:.2f}")
    # Save the updated JSON file
    with open(out_file, 'w') as f:
        json.dump(output, f)


process_pdb_folder(args.in_folder)
