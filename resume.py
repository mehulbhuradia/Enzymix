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


def plotter(c_0, c_denoised,c_noisy):
    c_0 = np.argmax(c_0.detach().to("cpu").numpy(), axis=1).reshape(-1, 1)
    c_denoised = np.argmax(c_denoised.detach().to("cpu").numpy(), axis=1).reshape(-1, 1)
    c_noisy = np.argmax(c_noisy.detach().to("cpu").numpy(), axis=1).reshape(-1, 1)

    # Plot Sequence
    plt.figure(figsize=(8, 6))
    plt.plot(c_0, color='r', label='true')
    # plt.plot(c_denoised, color='b', label='pred')
    plt.plot(c_noisy, color='g', label='noisy')
    plt.legend()
    plt.title('Sequence')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train.yml")
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
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

    args.resume="D:/Thesis/Enzymix/downloads/seq/positional_layers_2_add_layers_32/checkpoints/51.pt"
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


    def test_one(i):
        with torch.no_grad():
            model.eval()
            
            one_hot, edges, batch_size = dataset.__getitem__(i)
            coords=one_hot.unsqueeze(0).to(args.device)
            one_hot=one_hot.unsqueeze(0).to(args.device)
            edges=[edge.unsqueeze(0).to(args.device) for edge in edges]

            batch_size=torch.tensor([batch_size]).to(args.device)
            batch = dataset.batches[i]
            loss_dict, c_0, c_denoised,c_noisy, t = model(one_hot, edges,batch_size,analyse=True)
            res_len = int(one_hot.shape[1]/batch_size.item())


            if (torch.isnan(c_denoised).any().item()):
                print("c_denoised has nan")            

            c_denoised=c_denoised.squeeze(0)
            
            c_in = np.argmax(c_0.detach().to("cpu").numpy(), axis=1)
            c_out = np.argmax(c_denoised.detach().to("cpu").numpy(), axis=1)
            
            counte = 0
            for i in range(len(c_in)):
                if c_in[i] != c_out[i]:
                    counte += 1

            length = len(c_in)

            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            
            print(loss_dict,"Incorrect", counte,"Total", length,t)
            plotter(c_0, c_denoised,c_noisy)

            
            if not torch.isfinite(loss):
                print('NaN or Inf detected.')
        
    test_one(0)
    test_one(0)
    test_one(0)
    test_one(0)
    test_one(0)
    test_one(0)
    test_one(0)
