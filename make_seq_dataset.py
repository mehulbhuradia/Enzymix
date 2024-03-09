from pyfaidx import Fasta
import torch
import json
from egnn_clean import get_edges_batch
from tqdm import tqdm

BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def extract_id(header):
    return header.split('|')[1]

sequences = Fasta('uniprot_sprot.fasta', key_function=extract_id)

def sequence_to_one_hot(sequence, base_amino_acids=BASE_AMINO_ACIDS):
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

for protid, seq in tqdm(sequences.items(), desc="Processing Sequences"):
    try:
        sequence = seq[:].seq
        seq_len = len(sequence)
        if seq_len <= 300:
            save_dict = {}
            save_dict['UNIPROT_ID'] = protid
            save_dict['num_nodes'] = seq_len
            one_hot = sequence_to_one_hot(sequence)
            edges, _ = get_edges_batch(seq_len, 1)
            save_dict['one_hot'] = one_hot.tolist()
            save_dict['edges'] = [ten.tolist() for ten in edges]

            with open('./processed_seq/' + protid + '_tensors_' + str(save_dict['num_nodes']) + '.json', 'w') as file:
                json.dump(save_dict, file)
    except Exception as e:
        # Handle exceptions or print an error message if needed
        print(f"Error processing {protid}: {e}")
        continue