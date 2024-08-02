import os
import shutil
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from model.dpm import FullDPM
from data.af_db_batched import ProtienStructuresDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./train.yml')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tag', type=str, default='Both')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--layers', type=int, default=8)
    
    # Task options
    parser.add_argument('--only_ca', action='store_true', default=False)
    parser.add_argument('--num_steps', type=int, default=1000)

    # Configurations
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--min_len', type=int, default=50)
    parser.add_argument('--path', type=str, default='/tudelft.net/staff-umbrella/DIMA/swiss_p')
    parser.add_argument('--test_size', type=int, default=0)
    
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    
    # Update config based on args
    config.train.batch_size = args.batch_size
    config.train.max_len = args.max_len
    config.train.min_len = args.min_len
    config.train.path = args.path

    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Check if the model uses all atoms or only CA atoms
    if args.only_ca:
        x_dim = 3
        print('Using only CA atoms')
    else:
        x_dim = 9
        print('Using all 3 atoms')

    dataset = ProtienStructuresDataset(path=config.train.path, max_len=config.train.max_len, min_len=config.train.min_len,batch_size=config.train.batch_size,only_ca=args.only_ca,test_size=args.test_size)
    
    # start a new wandb run to track this script
    wandb_config = {"args": args,
                    "config": config,
                    "config_name": config_name,
                    "log_dir": log_dir,
                    "only_ca": args.only_ca,
                    "test_set": dataset.get_test_set(),
                    }
    wandb.init(
                project="DZYN",
                name=log_dir.split('/')[-1], # / for linux, \\ for windows
                config=wandb_config,
                mode="online" if args.wandb else "disabled",
            )
    
    print(args)
    print(config)
    print("Loading dataset...")
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
    print("Dataset size: ", dataset.size())
    print('Train Batches %d | Val Batches %d' % (len(train_dataset), len(val_dataset)))

    # Check if CUDA (GPU support) is available
    if not torch.cuda.is_available():
        print("No GPU available, using CPU.")
        args.device = 'cpu'

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
    
    # model.double()
    print('Number of parameters: %d' % count_parameters(model))
    print('Name of  the run: %s' % args.tag)
    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    
    # Resume
    if args.resume is not None:
        ckpt_path = args.resume
        print('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        print('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        print('Resuming from iteration %d' % it_first)

    # Train
    def train(it):
        
        model.train()
        avg_loss = {}
        avg_loss['overall'] = 0
        # avg_loss['rot'] = 0
        avg_loss['pos'] = 0
        avg_loss['seq'] = 0
        avg_forward_time = 0
        avg_backward_time = 0
        number_of_samples = len(train_loader)

        for i,x in enumerate(tqdm(train_loader, desc='Training Epoch: '+str(it), dynamic_ncols=True)):
            time_start = current_milli_time()
            x = recursive_to(x, args.device)
            loss_dict = model(x[0], x[1], x[2], x[3])
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()

            if not torch.isfinite(loss):
                print('NaN or Inf detected.')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'batch': recursive_to(x, 'cpu'),
                }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
                raise KeyboardInterrupt()
            
            # Backward
            loss.backward()
            time_backward_end = current_milli_time()

            avg_loss['overall'] += loss_dict['overall']
            # avg_loss['rot'] += loss_dict['rot']
            avg_loss['pos'] += loss_dict['pos']
            avg_loss['seq'] += loss_dict['seq']

            avg_forward_time += (time_forward_end - time_start)
            avg_backward_time += (time_backward_end - time_forward_end)

            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
        avg_loss['overall'] /= number_of_samples
        avg_loss['pos'] /= number_of_samples
        avg_loss['seq'] /= number_of_samples
        avg_forward_time /= number_of_samples
        avg_backward_time /= number_of_samples

        if avg_loss['seq'] > 0.7:
            if it == 1 and args.wandb:
                wandb.alert(title="High Seq Loss on start", text="Bad start for sequence, run will probably fail.")
            elif it > 10:
                print("High Seq Loss after 10 iters, stopping training")
                raise KeyboardInterrupt("High Seq Loss after 10 iters, stopping training")

        return avg_loss, avg_forward_time, avg_backward_time

    # Validate
    def validate(it):
        avg_loss = {}
        avg_loss['overall'] = 0
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
                # avg_loss['rot'] += loss_dict['rot']
                avg_loss['pos'] += loss_dict['pos']
                avg_loss['seq'] += loss_dict['seq']
        
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss['overall'])
        else:
            scheduler.step()

        avg_loss['overall'] /= number_of_samples
        avg_loss['pos'] /= number_of_samples
        avg_loss['seq'] /= number_of_samples

        return avg_loss

    # Main training loop
    try:
        # Set up early stopping
        early_stopping = {'counter': 0, 'best_loss': float('inf'), 'best_pos': float('inf'), 'best_seq': float('inf')}
        max_patience = config.train.early_stop_patience

        for it in range(it_first, config.train.max_epochs + 1):
            
            loss_dict, time_forward, time_backward = train(it)

            # Logging
            log_losses(loss_dict, it, 'train', others={
                # 'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_time_forward': (time_forward) / 1000,
                'avg_time_backward': (time_backward) / 1000,
            })

            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                # Logging
                log_losses(avg_val_loss, it, 'val')  
                
                # Check for early stopping
                # if avg_val_loss['overall'] < early_stopping['best_loss']:
                if avg_val_loss['pos'] < early_stopping['best_pos'] or avg_val_loss['seq'] < early_stopping['best_seq']:
                    # early_stopping['best_loss'] = avg_val_loss['overall']
                    early_stopping['best_pos'] = avg_val_loss['pos']
                    early_stopping['best_seq'] = avg_val_loss['seq']
                    early_stopping['counter'] = 0
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
                        print("Early stopping triggered")
                        break
    
    except KeyboardInterrupt:
        print('Terminating...')
