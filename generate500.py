import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
# tensorboard --logdir D:\Thesis\Enzymix\logs\   
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib.pyplot as plt
import numpy as np
import csv

from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from model.dpm import FullDPM
from data.af_db_batched import split_array
from visualisation.makepdb import create_pdb_file
from model.egnn_complex import get_edges_batch

BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_acids = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default="./logs/final/checkpoints/final.pt")
parser.add_argument('--layers', type=int, default=8)

# Task options
parser.add_argument('--only_ca', action='store_true', default=False)
parser.add_argument('--num_steps', type=int, default=1000)

# Configurations
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--min_len', type=int, default=50)


# Args only for sampling
parser.add_argument('--output', type=str, default='generated/')

args = parser.parse_args()

# Load configs
config, config_name = load_config('train.yml')
seed_all(config.train.seed)

# Update config based on args
config.train.max_len = args.max_len
config.train.min_len = args.min_len

# Check if the model uses all atoms or only CA atoms
if args.only_ca:
    x_dim = 3
    print('Using only CA atoms')
else:
    x_dim = 9
    print('Using all 3 atoms')


# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
else:
    print("No GPU available, using CPU.")

# Model
print('Building model...')
model = FullDPM(n_layers=args.layers,
                additional_layers=0,
                hidden_nf=1024,
                x_dim=x_dim,
                attention=True,
                normalize=True,
                residual=False,
                coords_agg='mean', # sum causes NaNs
                tanh=False,
                num_steps=args.num_steps
                    ).to(args.device)

# Resume
ckpt_path = args.resume
ckpt = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(ckpt['model'])


def sample_one(length):
    model.eval()
    batch_size = 1
    batch_size = torch.tensor([batch_size]).to(args.device)
    coords = torch.rand(1, length, 9).to(args.device)
    one_hot = torch.rand(1, length, 20).to(args.device)
    edges,_ = get_edges_batch(int(length/2), batch_size)
    edges=[edge.unsqueeze(0).to(args.device) for edge in edges]
    try:
        traj = model.sample(coords, one_hot, edges,batch_size, pbar=False, sample_structure=True, sample_sequence=True)
    except:
        return sample_one(length)
    return traj,length

def make_pdb(traj, res_len, global_counter,folder_path="generated/pdb",num=0,only_ca=False):
    sequences=traj[num][1].squeeze(0)
    sequence_list = split_array(sequences, res_len)
    positions=traj[num][0].detach().to("cpu").squeeze(0).numpy()
    positions_list = split_array(positions, res_len)

    # Make PDB file and fasta sequences
    fasta_content = ""
    for i in range(len(sequence_list)):
        sequence_name = []
        sequence_fasta = ""
        for j in sequence_list[i].tolist():
            sequence_name.append(amino_acids[j])
            sequence_fasta+=BASE_AMINO_ACIDS[j]
        residues=[]
        for j in range(len(sequence_name)):
            temp = {}
            temp['name'] = sequence_name[j]
            temp['CA'] = positions_list[i][j][:3].tolist()
            if not only_ca:
                temp['C'] = positions_list[i][j][3:6].tolist()
                temp['N'] = positions_list[i][j][6:].tolist()
            residues.append(temp)
        create_pdb_file(residues, folder_path+str(global_counter)+"_"+str(num)+".pdb")
        sequence_id = str(global_counter)+"_"+str(num)
        fasta_seq = split_array(sequence_fasta, 60)
        fasta_content += f">{sequence_id}"
        for fs in fasta_seq:
            fasta_content += f"\n{''.join(fs)}"
        fasta_content += "\n"
        global_counter+=1

    return fasta_content, global_counter


def generate(folder_path="generated/",min_len=50,max_len=100,only_ca=False):
    global_counter = 0
    fasta_content = ""
    # Create folder for generated PDB files
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(folder_path+"/pdb"):
        os.makedirs(folder_path+"/pdb")
    if not os.path.exists(folder_path+"/fasta"):
        os.makedirs(folder_path+"/fasta")
    for length in range(min_len,max_len):    
        print(f"Generating 10 sequences of length {length}")
        for sn in range(10):
            traj,res_len = sample_one(length=length)
            print(f"Generated sequence {sn+1} of length {res_len}")
            fasta_content_i,global_counter = make_pdb(traj, res_len, global_counter,folder_path=folder_path+"/pdb/",only_ca=only_ca)
            fasta_content += fasta_content_i
    fasta_filename = folder_path + "fasta/" + "generated.fasta"
    with open(fasta_filename, "w") as fasta_file:
        fasta_file.write(fasta_content)


generate(args.output,args.min_len,args.max_len,args.only_ca)

# Takes max 2 minutes per sequence
