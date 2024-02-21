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


def plotter(esp_pred, eps, c_0, c_denoised,p_noisy,p_0):

    c_0 = np.argmax(c_0.detach().to("cpu").numpy(), axis=1).reshape(-1, 1)
    c_denoised = np.argmax(c_denoised.detach().to("cpu").numpy(), axis=1).reshape(-1, 1)

    array1 = esp_pred.detach().to("cpu").squeeze(0).t().numpy()
    array2 = eps.detach().to("cpu").squeeze(0).t().numpy()
    array3 = p_noisy.detach().to("cpu").squeeze(0).t().numpy()
    array4 = p_0.detach().to("cpu").squeeze(0).t().numpy()
    
    
    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot each line in a subplot
    # g_CA_coords
    axs[0][0].plot(array1[0], color='r', label='pred')
    axs[0][0].plot(array2[0], color='b', label='eps')
    # axs[0][0].plot(array3[0], color='g', label='p_noisy')
    # axs[0][0].plot(array4[0], color='y', label='p_0')
    
    # Add legend
    axs[0][0].legend()
    axs[0][0].set_title('C-a x')
    # g_CA_coords
    axs[1][0].plot(array1[1], color='r', label='pred')
    axs[1][0].plot(array2[1], color='b', label='eps')
    # axs[1][0].plot(array3[1], color='g', label='p_noisy')
    # axs[1][0].plot(array4[1], color='y', label='p_0')
    
    # Add legend
    axs[1][0].legend()
    axs[1][0].set_title('C-a y')
    # g_CA_coords
    axs[0][1].plot(array1[2], color='r', label='pred')
    axs[0][1].plot(array2[2], color='b', label='eps')
    # axs[0][1].plot(array3[2], color='g', label='p_noisy')
    # axs[0][1].plot(array4[2], color='y', label='p_0')

    # Add legend
    axs[0][1].legend()
    axs[0][1].set_title('C-a z')
    # sequence
    axs[1][1].plot(c_0, color='r', label='pred')
    axs[1][1].plot(c_denoised, color='b', label='true')
    # Add legend
    axs[1][1].legend()
    axs[1][1].set_title('Sequence')

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
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--add_layers', type=int, default=0)
    parser.add_argument('--uni', type=str, default=None)


    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    args.resume="D:/Thesis/Enzymix/hybridmodel_logs/logs/40_3/checkpoints/280.pt"
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


    def test_one(uniprotid,t):
        with torch.no_grad():
            model.eval()
            
            coords, one_hot, edges, path = dataset.get_item_by_uniprotid(uniprotid)
            
            coords=coords.unsqueeze(0).to(args.device)
            one_hot=one_hot.unsqueeze(0).to(args.device)
            edges=[edge.unsqueeze(0).to(args.device) for edge in edges]

            
            loss_dict, eps_pred, eps_p, c_0, c_denoised,t,p_noisy,p_0 = model(coords, one_hot, edges,analyse=True,t=t)
            
            if (torch.isnan(eps_p).any().item()):
                print("eps_p has nan")

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
            
            # if "A0A1D6H1J3" in path:
            # if args.uni in path:
            print(loss_dict,"Incorrect", counte,"Total", length,path,t)

            # if "A0A1D6H1J3" in path:
            plotter(eps_pred, eps_p, c_0, c_denoised,p_noisy,p_0)

            
            if not torch.isfinite(loss):
                print('NaN or Inf detected.')
        
    # if args.uni:
    #     test_one(args.uni)
    # else:
    #     for i in dataset.paths:
    t = torch.randint(1, 10, (1,), dtype=torch.long)
    test_one(dataset.paths[0],t)
    t = torch.randint(45, 55, (1,), dtype=torch.long)
    test_one(dataset.paths[0],t)
    t = torch.randint(90, 100, (1,), dtype=torch.long)
    test_one(dataset.paths[0],t)
    

