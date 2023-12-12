#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --job-name=Enzymix
#SBATCH --qos=medium         
#SBATCH --time=48:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use ../../opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 # Load certain versions of cuda and cudnn 

pip install torch torchvision torchaudio
pip install tqdm easydict tensorboard torch_geometric graphein

srun python train.py train.yml