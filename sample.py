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

BASE_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


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
    parser.add_argument('--add_layers', type=int, default=32)
    parser.add_argument('--uni', type=str, default=None)

    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    args.resume="D:/Thesis/Enzymix/logs/test/checkpoints/47.pt"
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
    dataset = ProtienStructuresDataset(path=config.train.path, max_len=config.train.max_len, min_len=config.train.min_len, batch_size=config.train.batch_size)
    d_loader = DataLoader(
        dataset, 
        batch_size=1,  
        shuffle=True,
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

def split_array(input_array, chunk_size):
  return [input_array[i:i + chunk_size] for i in range(0, len(input_array), chunk_size)]


def sample_one(i):
    model.eval()
    print(f"Sampling from {i}")
    one_hot, edges= dataset.__getitem__(i)
    one_hot=one_hot.unsqueeze(0).to(args.device)
    edges=[edge.unsqueeze(0).to(args.device) for edge in edges]
    batch = dataset.batches[i]
    traj = model.sample(one_hot, edges)
    res_len = int(batch[0].split('/')[2].split('_')[2].split('.')[0])

    seq_gen=""
    for j in range(0, len(traj[0][0])):
        seq_gen+=BASE_AMINO_ACIDS[traj[0][0][j]]

    seqs = split_array(seq_gen, res_len)

    fasta_content = ""
    for k in range(len(seqs)):
        sequence_id = f"sequence_{i}_{k}"
        fasta_seq = split_array(seqs[k], 60)
        fasta_content += f">{sequence_id}"
        for fs in fasta_seq:
            fasta_content += f"\n{''.join(fs)}"
        fasta_content += "\n"
    return fasta_content

fasta_content = ""
for i in range(0, len(dataset)):
    fasta_content += sample_one(i)

fasta_filename = f"generated_seqs/output.fasta"    
with open(fasta_filename, "w") as fasta_file:
    fasta_file.write(fasta_content)

print(f"Generated {fasta_filename}")


