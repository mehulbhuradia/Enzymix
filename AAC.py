# load fasta fiel

import sys
import os

from pyfaidx import Fasta
sequences = Fasta("generated/fasta/generated.fasta")
seq_ids = list(sequences.keys())
seqs={}
for seq in sequences:
    seqs[seq[:].name] = [seq[:].seq]

# Load all fasta files from ./proteinmpnn_residues
fasta_files = os.listdir("./proteinmpnn_residues")
# only files that end with .fasta
fasta_files = [file for file in fasta_files if file.endswith(".fasta")]

for file in fasta_files:
    for id in seqs.keys():
        if file.split("_")[0] == id.split("_")[0]:
            pmpnn_sequence = Fasta(f"./proteinmpnn_residues/{file}")
            for seq in pmpnn_sequence:
                seqs[id].append(seq[:].seq)
            
average_percent = 0
percent_dist = {}
for id, s_list in seqs.items():
    og = s_list[0]
    all_percent = []
    for i in range(1,len(s_list)):
        total = 0
        correct = 0
        print(id)
        print(len(s_list))
        print(len(og), len(s_list[i]))
        print(s_list)
        for j in range(len(s_list[i])):
            if og[j] == s_list[i][j]:
                correct += 1
            total += 1
        percent = correct/total*100
        all_percent.append(percent)
    max_percent = max(all_percent)
    average_percent += max_percent
    # print(f"{id}: {max_percent}")
    percent_dist[id] = max_percent

average_percent = average_percent/len(seqs)
print(f"Average: {average_percent}")

# Save dict to json
with open("percent_dist.json", "w") as f:
    f.write(str(percent_dist))