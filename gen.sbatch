#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=gen
#SBATCH --qos=short         
#SBATCH --time=4:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

export HOME=/tmp/mbhuradia/

srun python generate500.py --resume /tudelft.net/staff-umbrella/DIMA/Enzymix/logs/final/checkpoints/final.pt