#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=F_Enzymix
#SBATCH --qos=long         
#SBATCH --time=168:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8   
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py --wandb --tag af50_16 --min_len 95 --path /tudelft.net/staff-umbrella/DIMA/Enzymix/af_50_processed --layers 16
