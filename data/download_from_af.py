import json
import argparse
import os
import requests
import time

parser = argparse.ArgumentParser(description='Download structures from AlphaFold')
parser.add_argument('--start', type=int, default=0, help='Start index')
parser.add_argument('--chunk', type=int, default=10, help='Chunk size')
parser.add_argument('--input', type=str, default='data/final_reps.json', help='Input file')
parser.add_argument('--output', type=str, default='./af_50', help='Output directory')
args = parser.parse_args()


if not os.path.exists(args.output):
    os.makedirs(args.output)


def download_alphafold_structure(uniprot_id, save_dir):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(url)
    if response.ok:
        filename = f"{save_dir}/{uniprot_id}.pdb"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"PDB file for UniProt ID {uniprot_id} downloaded successfully to {save_dir}.")
        return filename
    else:
        print(f"Failed to download PDB file for UniProt ID {uniprot_id}.")
        return None

with open(args.input, 'r') as file:
    data = json.load(file)

ids=[]
for prot in data:
    ids.append(prot)

count=0
fails=[]
for protid in ids[args.start:args.start+args.chunk]:
    protein_path = download_alphafold_structure(protid, args.output)
    if protein_path:
        count+=1
    else:
        fails.append(protid)
    time.sleep(1)

print("Downloaded ", count, " structures.")
print("Failed to download ", len(fails), " structures.")
print("Failed structures: ", fails)

if len(fails)>0:
    with open('failed_'+str(args.start)+'.json', 'w') as file:
        json.dump(fails, file)