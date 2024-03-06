from pyfaidx import Fasta
import torch
import json

with open('data_water.json', 'r') as file:
    data = json.load(file)

with open('data.json', 'r') as file:
    data.extend(json.load(file))

ids=set()
for prot in data:
    ids.add(prot['uniprot'])
    
print(len(ids))

BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def extract_id(header):
    return header.split('|')[1]

sequences = Fasta('uniprot_sprot.fasta', key_function=extract_id)

def uniprot_to_one_hot(uniprot, base_amino_acids=BASE_AMINO_ACIDS):
    sequence = sequences[uniprot][:].seq
    one_hot_encoding = []
    
    for amino_acid in sequence:
        if amino_acid in base_amino_acids:
            # Create a binary vector for each amino acid
            one_hot_vector = [1 if amino_acid == aa else 0 for aa in base_amino_acids]
            one_hot_encoding.append(one_hot_vector)
        else:
            print(f'Invalid amino acid: {amino_acid}')
            raise ValueError(f'Invalid amino acid: {amino_acid}')

    return torch.tensor(one_hot_encoding, dtype=torch.float32)


# count=0
# for protid in list(ids):
#     protein_path = download_alphafold_structure(protid, out_dir = "./af_structures_mmcif", aligned_score=False,pdb=False,version= 4,mmcif=True)
#     count += 1

# print(count)

print(uniprot_to_one_hot('P33197').shape)
