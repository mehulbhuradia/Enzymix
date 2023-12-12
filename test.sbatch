#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --job-name=Enzymix
#SBATCH --qos=short         
#SBATCH --time=00:10:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 # Load certain versions of cuda and cudnn 

pip3 install torch torchvision torchaudio
pip3 install tqdm easydict tensorboard torch_geometric graphein

srun python train.py train_dummy.yml