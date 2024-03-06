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
from af_db import ProtienStructuresDataset
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train.yml")
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')

    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--add_layers', type=int, default=0)
    parser.add_argument('--uni', type=str, default=None)


    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    args.resume="D:/Thesis/Enzymix/atom_logs/logs/train_2024_02_27__02_59_00atoms_layers_40_add_layers_0/checkpoints/35.pt"
    # Logging
    if args.debug:
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag) + args.name +'_layers_'+str(args.layers)+'_add_layers_'+str(args.add_layers)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    print(args)
    print(config)

    # Data
    print('Loading dataset...')
    dataset = ProtienStructuresDataset(path=config.train.path, max_len=config.train.max_len)
    d_loader = DataLoader(
        dataset, 
        batch_size=1,  
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Samples %d' % (len(dataset)))

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
    else:
        print("No GPU available, using CPU.")

    # Model
    print('Building model...')
    model = FullDPM(n_layers=args.layers,additional_layers=args.add_layers).to(args.device)

    # Resume
    ckpt_path = args.resume
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])


def sample_one(i):
    model.eval()
    print(f"Sampling from {i}")
    one_hot, edges= dataset.__getitem__(i)
    one_hot=one_hot.unsqueeze(0).to(args.device)
    edges = edges.unsqueeze(0).to(args.device) 
    traj = model.sample(one_hot, edges)
    return traj,model.trans_seq._sample(one_hot).squeeze(0)


def make_pdb(traj,num=0):
    sequence_0=traj[num][1].squeeze(0)
    sequence_0,_,_=torch.chunk(sequence_0, chunks=3, dim=-1)
    position_0=traj[num][0]
    g_CA_coords,g_C_coords,g_N_coords = torch.chunk(position_0, chunks=3, dim=-2)
    position_0 = torch.cat((g_CA_coords,g_C_coords,g_N_coords),dim=-1)
    position_0 = position_0.detach().to("cpu").squeeze(0).numpy()

    sequence_0_name = []
    for i in sequence_0.tolist():
        sequence_0_name.append(amino_acids[i])

    residues_0=[]
    for i in range(len(sequence_0_name)):
        temp = {}
        temp['name'] = sequence_0_name[i]
        temp['CA'] = position_0[i][:3].tolist()
        temp['CB'] = position_0[i][3:6].tolist()
        temp['CN'] = position_0[i][6:].tolist()
        residues_0.append(temp)
    create_pdb_file(residues_0, "traj/"+str(num)+".pdb")
    

traj,coords,s_true = sample_one(0)
for i in range(len(traj)):
    make_pdb(traj,i)

