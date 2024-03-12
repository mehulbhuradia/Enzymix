from pyfaidx import Fasta
from tqdm import tqdm

BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def split_array(input_array, chunk_size):
  return [input_array[i:i + chunk_size] for i in range(0, len(input_array), chunk_size)]

def extract_id(header):
    return header.split('|')[1]

sequences = Fasta('uniprot_sprot.fasta', key_function=extract_id)


valid_sequences = []
for protid, seq in tqdm(sequences.items(), desc="Processing Sequences"):
    sequence = seq[:].seq
    for amino_acid in sequence:
        if amino_acid not in BASE_AMINO_ACIDS:
            print(f'Invalid amino acid: {amino_acid}')
            continue
    if len(sequence) <= 200 and len(sequence) >= 50:
        valid_sequences.append(sequence)

# randomly pick 2500 sequences
import random
random.seed(42)
random.shuffle(valid_sequences)
valid_sequences = valid_sequences[:2500]

fasta_content = ""
for k in range(len(valid_sequences)):
    sequence_id = f"sequence_{k}"
    fasta_seq = split_array(valid_sequences[k], 60)
    fasta_content += f">{sequence_id}"
    for fs in fasta_seq:
        fasta_content += f"\n{''.join(fs)}"
    fasta_content += "\n"

fasta_filename = f"generated_seqs/og.fasta"    
with open(fasta_filename, "w") as fasta_file:
    fasta_file.write(fasta_content)

