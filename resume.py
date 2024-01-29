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

from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from dpm import FullDPM
from af_db import ProtienStructuresDataset

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


    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    args.resume="D:/Thesis/Enzymix/logs_big_11_1_24/EGCL_2_MLP_24/checkpoints/3460.pt"
    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag) + args.name +'_layers_'+str(args.layers)+'_add_layers_'+str(args.add_layers)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    dataset = ProtienStructuresDataset(path=config.train.path, max_len=config.train.max_len)
    d_loader = DataLoader(
        dataset, 
        batch_size=1,  
        shuffle=False,
        num_workers=args.num_workers
    )
    logger.info('Samples %d' % (len(dataset)))

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
    else:
        logger.info("No GPU available, using CPU.")

    # Model
    logger.info('Building model...')
    model = FullDPM(n_layers=args.layers,additional_layers=args.add_layers).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))
    logger.info('Number of EGCL layers: %d' % args.layers)
    logger.info('Number of additional layers: %d' % args.add_layers)
    logger.info('Name of  the run: %s' % args.name)
    

    # Resume
    ckpt_path = args.resume
    logger.info('Resuming from checkpoint: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])


    for i,x in enumerate(tqdm(d_loader, desc='Running: ', dynamic_ncols=True)):
        time_start = current_milli_time()
        x = recursive_to(x, args.device)
        print(x[4])
        # Forward
        # if args.debug: torch.set_anomaly_enabled(True)

        loss_dict, p_pred, p_0, c_0, c_denoised = model(x[0], x[1], x[2], x[3],analyse=True)

        array1 = p_pred.detach().to("cpu").squeeze(0).t().numpy()
        array2 = p_0.detach().to("cpu").squeeze(0).t().numpy()
        
        print(array1[0][0])
        print(array2[0][0])
        print(array1[0][1])
        print(array2[0][1])
        print(array1[0][2])
        print(array2[0][2])

        # Create subplots
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))

        # Plot each line in a subplot
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                axs[i, j].plot(array1[idx], color='r', label='pred')
                axs[i, j].plot(array2[idx], color='b', label='p_0')

        # Show the plot
        plt.show()
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()


        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            raise KeyboardInterrupt()
            


    