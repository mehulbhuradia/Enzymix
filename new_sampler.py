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
from dpm import FullDPM
from af_db_batched import ProtienStructuresDataset, split_array
from makepdb import create_pdb_file

amino_acids=["ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PYL",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "SEC",
    "VAL",
    "TRP",
    "TYR",
    "UNK"
]





parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default=None)

# Task options
parser.add_argument('--only_ca', action='store_true', default=False)
parser.add_argument('--num_steps', type=int, default=100)

# EGNN options
parser.add_argument('--eg_attention', action='store_true', default=False)
parser.add_argument('--eg_aggregate', type=str, default='mean') # mean or sum
parser.add_argument('--eg_disable_residual', action='store_false', default=True)
parser.add_argument('--eg_normalize', action='store_true', default=False)
parser.add_argument('--eg_tanh', action='store_true', default=False)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--add_layers', type=int, default=0)
parser.add_argument('--node_features', type=int, default=1024)

# Configurations
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--min_len', type=int, default=50)

args = parser.parse_args()

log_dir = "./logs/train_2024_04_05__04_00_52baseline_layers_4_add_layers_0_node_features_1024/"
checkpoint = "120.pt"

# Load configs
config, config_name = load_config('train.yml')
seed_all(config.train.seed)
# Update config based on args
config.train.batch_size = args.batch_size
config.train.max_len = args.max_len
config.train.min_len = args.min_len

args.resume=log_dir + 'checkpoints/' + checkpoint

# Check if the model uses all atoms or only CA atoms
if args.only_ca:
    x_dim = 3
    print('Using only CA atoms')
else:
    x_dim = 9
    print('Using all 3 atoms')

# Data
print('Loading dataset...')
dataset = ProtienStructuresDataset(path=config.train.path, max_len=config.train.max_len, min_len=config.train.min_len,batch_size=config.train.batch_size,only_ca=args.only_ca)
print("Dataset size: ", dataset.size())
print('Samples %d' % (len(dataset)))

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
else:
    print("No GPU available, using CPU.")

# Model
print('Building model...')
model = FullDPM(n_layers=args.layers,
        additional_layers=args.add_layers,
        hidden_nf=args.node_features,
        x_dim=x_dim,
        attention=args.eg_attention,
        normalize=args.eg_normalize,
        residual=args.eg_disable_residual,
        coords_agg=args.eg_aggregate,
        tanh=args.eg_tanh,
        num_steps=args.num_steps
            ).to(args.device)

# Resume
ckpt_path = args.resume
ckpt = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(ckpt['model'])


def sample_one(idx):
    model.eval()
    print(f"Sampling from {idx}")
    coords, one_hot, edges, batch_size = dataset[idx]
    res_len = coords.shape[0]//batch_size
    coords=coords.unsqueeze(0).to(args.device)
    one_hot=one_hot.unsqueeze(0).to(args.device)
    edges=[edge.unsqueeze(0).to(args.device) for edge in edges]
    batch_size = torch.tensor([batch_size]).to(args.device)
    traj = model.sample(coords, one_hot, edges,batch_size, pbar=True, sample_structure=True, sample_sequence=True)
    return traj,batch_size,res_len


def make_pdb(traj, res_len, global_counter,num=0):
    sequences=traj[num][1].squeeze(0)
    sequence_list = split_array(sequences, res_len)
    positions=traj[num][0].detach().to("cpu").squeeze(0).numpy()
    positions_list = split_array(positions, res_len)

    for i in range(len(sequence_list)):
        sequence_name = []
        for j in sequence_list[i].tolist():
            sequence_name.append(amino_acids[j])
        residues=[]
        for j in range(len(sequence_name)):
            temp = {}
            temp['name'] = sequence_name[j]
            temp['CA'] = positions_list[i][j][:3].tolist()
            temp['CB'] = positions_list[i][j][3:6].tolist()
            temp['CN'] = positions_list[i][j][6:].tolist()
            residues.append(temp)
        # check if the directory exists
        if not os.path.exists("generated"):
            os.makedirs("generated")
        create_pdb_file(residues, "generated/"+str(global_counter)+"_"+str(num)+".pdb")
        global_counter+=1
    return global_counter

global_counter = 0
for i in range(10):
    traj,batch_size,res_len = sample_one(i)
    global_counter = make_pdb(traj, res_len, global_counter)

