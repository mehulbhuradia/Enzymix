import csv
import os
from tqdm import tqdm


from pyfaidx import Fasta


file_path = "./extras/1-AFDBClusters-entryId_repId_taxId.tsv"

path = './swiss_p'
max_len = 100
min_len = 50

ids = []

for pdb in os.listdir(path):
    length=pdb.split('_')[2].split('.')[0]
    if int(length) <= max_len and int(length) >= min_len:
        id = pdb.split('_')[0]
        ids.append(id)


print(f"Number of ids: {len(ids)}")

tracker ={}



# Open the TSV file in read mode with appropriate encoding
with open(file_path, 'r', encoding='utf-8') as tsvfile:
    # Use csv reader to read the TSV file, specifying the delimiter as '\t' (tab)
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    
    # Iterate through each row in the TSV file
    for row in tsvreader:
        # Print each row
        memberID = row[0]
        repId = row[1]
        if repId not in tracker:
            tracker[repId] = {'size': 1, 'members': []}
        else:
            tracker[repId]['size'] += 1
        tracker[repId]['members'].append(memberID)


# min size
minimum = min(tracker, key=lambda x: tracker[x]['size'])
print(f"Minimum size: {minimum['size']}")

# min_val = min(tracker.values())
number_of_clusters = len(tracker)

print(f"Number of clusters: {number_of_clusters}")
# print(f"Minimum number of members in a cluster: {min_val}")


# BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# def extract_id(header):
#     return header.split('|')[1]

# sequences = Fasta('./extras/uniprot_sprot.fasta', key_function=extract_id)


# for protid, seq in tqdm(sequences.items(), desc="Processing Sequences"):
#     sequence = seq[:].seq
#     seq_len = len(sequence)



# count = 0
# for key in tqdm(tracker):
#     if key in ids:
#         count += 1

# print(f"Number of clutsers in dataset: {count}")


# Number of clutsers in dataset: 427
