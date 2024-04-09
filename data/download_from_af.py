import json
from graphein.protein.utils import download_alphafold_structure
import argparse
import os

parser = argparse.ArgumentParser(description='Download structures from AlphaFold')
parser.add_argument('--start', type=int, default=0, help='Start index')
parser.add_argument('--chunk', type=int, default=500, help='Chunk size')
parser.add_argument('--input', type=str, default='data/final_reps.json', help='Input file')
parser.add_argument('--output', type=str, default='./af_50', help='Output directory')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(args.input, 'r') as file:
    data = json.load(file)

ids=set()
for prot in data:
    ids.add(prot)

count=0
for protid in list(ids)[args.start:args.start+args.chunk]:
    protein_path = download_alphafold_structure(protid, out_dir = args.output, aligned_score=False,version= 4)
    count += 1

print("Downloaded ", count, " structures.")