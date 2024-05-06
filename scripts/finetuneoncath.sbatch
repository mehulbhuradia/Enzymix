#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=cath
#SBATCH --qos=long
#SBATCH --time=168:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py --wandb --tag cathfinetune --resume ./logs/cath/checkpoints/final.pt --num_steps 1000 --path ./cath_train --test_size 0
