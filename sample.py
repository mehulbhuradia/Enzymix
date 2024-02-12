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



def plotter(t100, t0, coords):

    array1 = t100.detach().to("cpu").squeeze(0).t().numpy()
    array2 = t0.detach().to("cpu").squeeze(0).t().numpy()
    array3 = coords.detach().to("cpu").squeeze(0).t().numpy()

    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot each line in a subplot
    # g_CA_coords
    axs[0][0].plot(array1[0], color='r', label='100')
    axs[0][0].plot(array2[0], color='b', label='0')
    axs[0][0].plot(array3[0], color='g', label='true')
    # Add legend
    axs[0][0].legend()
    axs[0][0].set_title('C-a x')
    # g_CA_coords
    axs[1][0].plot(array1[1], color='r', label='100')
    axs[1][0].plot(array2[1], color='b', label='0')
    axs[1][0].plot(array3[1], color='g', label='true')
    # Add legend
    axs[1][0].legend()
    axs[1][0].set_title('C-a y')
    # g_CA_coords
    axs[0][1].plot(array1[2], color='r', label='100')
    axs[0][1].plot(array2[2], color='b', label='0')
    axs[0][1].plot(array3[2], color='g', label='true')
    # Add legend
    axs[0][1].legend()
    axs[0][1].set_title('C-a z')
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train.yml")
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')

    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--add_layers', type=int, default=24)
    parser.add_argument('--uni', type=str, default=None)


    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    args.resume="D:/Thesis/Enzymix/logs/noise_100_2_24_2000/checkpoints/2000.pt"
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


def sample_one(uniprotid):
    model.eval()
    print(f"Sampling from {uniprotid}")
    coords, one_hot, edges, path = dataset.get_item_by_uniprotid(uniprotid)
    coords=coords.unsqueeze(0).to(args.device)
    one_hot=one_hot.unsqueeze(0).to(args.device)
    edges=[edge.unsqueeze(0).to(args.device) for edge in edges]

    traj = model.sample(coords, one_hot, edges, pbar=True)
    return traj,coords


if args.uni:
    sample_one(args.uni)
else:
    for i in dataset.paths:
        traj,coords = sample_one(i)
        break

sequence_100=traj[100][1].squeeze(0)
position_100=traj[100][0]
sequence_0=traj[0][1].squeeze(0)
position_0=traj[0][0]
                 

plotter(position_100, position_0, coords)


position_0 = position_0.detach().to("cpu").squeeze(0).numpy()
position_100 = position_100.detach().to("cpu").squeeze(0).numpy()



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

sequence_0_name = []
for i in sequence_0.tolist():
    sequence_0_name.append(amino_acids[i])

sequence_100_name = []
for i in sequence_100.tolist():
    sequence_100_name.append(amino_acids[i])

residues_100=[]
for i in range(len(sequence_100_name)):
    temp = {}
    temp['name'] = sequence_100_name[i]
    temp['CA'] = position_100[i][:3].tolist()
    temp['CB'] = position_100[i][3:6].tolist()
    temp['CN'] = position_100[i][6:].tolist()
    residues_100.append(temp)

from makepdb import create_pdb_file

create_pdb_file(residues_100, "100.pdb")

residues_0=[]
for i in range(len(sequence_0_name)):
    temp = {}
    temp['name'] = sequence_0_name[i]
    temp['CA'] = position_0[i][:3].tolist()
    temp['CB'] = position_0[i][3:6].tolist()
    temp['CN'] = position_0[i][6:].tolist()
    residues_0.append(temp)

create_pdb_file(residues_0, "0.pdb")

residues_true=[]
coords = coords.detach().to("cpu").squeeze(0).numpy()
for i in range(len(sequence_0_name)):
    temp = {}
    temp['name'] = sequence_0_name[i]
    temp['CA'] = coords[i][:3].tolist()
    temp['CB'] = coords[i][3:6].tolist()
    temp['CN'] = coords[i][6:].tolist()
    residues_true.append(temp)

create_pdb_file(residues_true, "true.pdb")