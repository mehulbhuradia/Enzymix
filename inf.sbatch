#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=F_Enzymix
#SBATCH --qos=infinite         
#SBATCH --time=1024:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py --name inf_8_0_1024_smallbatch --layers 8 --add_layers 0 --node_features 1024 --wandb --batch_size 1
