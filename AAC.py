# load fasta fiel

import sys
import os
import numpy as np

from pyfaidx import Fasta
sequences = Fasta("generated_final/fasta/generated.fasta")
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
percent_list = []
percent_dist = {}
for id, s_list in seqs.items():
    og = s_list[0]
    total = 0
    correct = 0
    for j in range(len(s_list[1])):
        if og[j] == s_list[1][j]:
            correct += 1
        total += 1
    percent = correct/total*100
    print(f"{id}: {percent}")
    average_percent += percent
    percent_dist[id] = percent
    percent_list.append(percent)

average_percent = np.mean(percent_list)
std_percent = np.std(percent_list)
print(f"Average: {average_percent}")
print(f"Standard deviation: {std_percent}")


# Save dict to json
with open("percent_dist.json", "w") as f:
    f.write(str(percent_dist))

# Histogram of percent_list
import matplotlib.pyplot as plt
plt.hist(percent_list, bins=10)
plt.xlabel('Amino Acid Concistency (%)')
plt.ylabel("Frequency")
plt.title('AAC of the generated protiens')
plt.savefig('./saved_plots/AAC_hist.png')
plt.show()

import json
with open('cctm_scores.json', 'r') as f:
    tm_dict = json.load(f)

tm_scores_list = list(tm_dict.values())
tm_scores_list = [float(i) for i in tm_scores_list]
tm_scores_list.sort(key = float)

# Make a scatter plot of values from tm_dict and percent_dist
# Match the values by the keys
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
for key in tm_dict.keys():
    if key in percent_dist.keys():
        x.append(tm_dict[key])
        y.append(percent_dist[key])


