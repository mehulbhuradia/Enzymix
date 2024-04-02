#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=B_CA_Enzymix
#SBATCH --qos=long         
#SBATCH --time=168:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py train.yml --name ca_both --layers $1 --add_layers $2 --node_features $3 --wandb --only_ca
