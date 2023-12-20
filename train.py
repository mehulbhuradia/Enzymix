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


from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from dpm import FullDPM
from af_db import ProtienStructuresDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag) + args.name
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
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
    else:
        logger.info("No GPU available, using CPU.")

    # Model
    logger.info('Building model...')
    model = FullDPM(n_layers=config.train.num_layers).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        ckpt_path = args.resume
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    # Train
    def train(it):
        
        model.train()
        avg_loss = {}
        avg_loss['overall'] = 0
        avg_loss['rot'] = 0
        avg_loss['pos'] = 0
        avg_loss['seq'] = 0
        avg_forward_time = 0
        avg_backward_time = 0
        number_of_samples = len(train_loader)

        for i,x in enumerate(tqdm(train_loader, desc='Training Epoch: '+str(it), dynamic_ncols=True)):
            time_start = current_milli_time()
            x = recursive_to(x, args.device)

            # Forward
            # if args.debug: torch.set_anomaly_enabled(True)

            loss_dict = model(x[0], x[1], x[2], x[3])
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()


            if not torch.isfinite(loss):
                logger.error('NaN or Inf detected.')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
                raise KeyboardInterrupt()
            
            # Backward
            loss.backward()
            time_backward_end = current_milli_time()

            avg_loss['overall'] += loss_dict['overall']
            avg_loss['rot'] += loss_dict['rot']
            avg_loss['pos'] += loss_dict['pos']
            avg_loss['seq'] += loss_dict['seq']

            avg_forward_time += (time_forward_end - time_start)
            avg_backward_time += (time_backward_end - time_forward_end)

            if i % config.train.batch_size == 0 or i == len(train_loader) - 1:
                # no gradent clipping
                # orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
        avg_loss['overall'] /= number_of_samples
        avg_loss['rot'] /= number_of_samples
        avg_loss['pos'] /= number_of_samples
        avg_loss['seq'] /= number_of_samples
        avg_forward_time /= number_of_samples
        avg_backward_time /= number_of_samples

        return avg_loss, avg_forward_time, avg_backward_time

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        avg_loss = {}
        avg_loss['overall'] = 0
        avg_loss['rot'] = 0
        avg_loss['pos'] = 0
        avg_loss['seq'] = 0
        number_of_samples = len(val_loader)
        with torch.no_grad():
            model.eval()
            for x in tqdm(val_loader, desc='Validate', dynamic_ncols=True):
                # Prepare data
                x = recursive_to(x, args.device)
                # Forward
                loss_dict = model(x[0], x[1], x[2], x[3])
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                # Accumulate
                avg_loss['overall'] += loss_dict['overall']
                avg_loss['rot'] += loss_dict['rot']
                avg_loss['pos'] += loss_dict['pos']
                avg_loss['seq'] += loss_dict['seq']
        
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss['overall'])
        else:
            scheduler.step()

        avg_loss['overall'] /= number_of_samples
        avg_loss['rot'] /= number_of_samples
        avg_loss['pos'] /= number_of_samples
        avg_loss['seq'] /= number_of_samples

        return avg_loss

    # Main training loop
    try:
        # Set up early stopping
        early_stopping = {'counter': 0, 'best_loss': float('inf')}
        max_patience = config.train.early_stop_patience

        for it in range(it_first, config.train.max_epochs + 1):
            
            loss_dict, time_forward, time_backward = train(it)

            # Logging
            log_losses(loss_dict, it, 'train', logger, writer, others={
                # 'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_time_forward': (time_forward) / 1000,
                'avg_time_backward': (time_backward) / 1000,
            })

            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                # Logging
                log_losses(avg_val_loss, it, 'val', logger, writer)  
                
                # Check for early stopping
                if avg_val_loss['overall'] < early_stopping['best_loss']:
                    early_stopping['best_loss'] = avg_val_loss['overall']
                    early_stopping['counter'] = 0
                    if not args.debug:
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                            'avg_val_loss': avg_val_loss['overall'],
                        }, ckpt_path)
                else:
                    early_stopping['counter'] += 1
                    if early_stopping['counter'] >= max_patience:
                        logger.info("Early stopping triggered")
                        break
    
    except KeyboardInterrupt:
        logger.info('Terminating...')
