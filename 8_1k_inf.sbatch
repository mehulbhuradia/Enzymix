#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=8_1k_inf
#SBATCH --qos=infinite         
#SBATCH --time=600:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py --wandb --tag 81kinf --resume ./logs/8_1k_inf/checkpoints/385.pt --num_steps 1000
