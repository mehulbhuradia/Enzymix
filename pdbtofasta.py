from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB import Polypeptide
from data.af_db_batched import split_array


def pdb_to_seq_str(pdb_file,name):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', pdb_file)

    # Get the sequence of the protein
    fasta_content = ""
    seq=""
    for model in structure:
        for chain in model:
            for residue in chain:
                print(residue)
                # CHECK IF IT WORKS< THERE WAS A BUG IN AMINO ACIDS WHILE ENERTING SO THIS IS ROBABLY CORRECT BUT CHECK BEFORE RUNNING
                ###################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                #####################################################################################################
                ##################################################
                # if PDB.is_aa(residue):
                one_letter_code = Polypeptide.three_to_index(residue.get_resname())
                one_letter_code = Polypeptide.index_to_one(one_letter_code)
                seq += one_letter_code
    print(len(seq))
    fasta_seq = split_array(seq, 60)
    fasta_content += f">{name}"
    for fs in fasta_seq:
        fasta_content += f"\n{''.join(fs)}"
    fasta_content += "\n"
    return fasta_content

import os
# List of files in the folder
folder_path = "./generated/pdb"
file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdb')]
file_list = sorted(file_list, key=lambda x: int(x.split("_")[0]))
fasta_content = ""
for filename in file_list:
    pdb_file = os.path.join(folder_path, filename)
    fasta_content_i = pdb_to_seq_str(pdb_file, filename.split(".")[0])
    fasta_content += fasta_content_i

# fasta_filename = "generated.fasta"
# with open(fasta_filename, "w") as fasta_file:
#     fasta_file.write(fasta_content)