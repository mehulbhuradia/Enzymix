#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=sample
#SBATCH --qos=medium         
#SBATCH --time=36:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

export HOME=/tmp/mbhuradia/
export HF_HOME=/tmp/mbhuradia/

srun python generate500.py --resume /tudelft.net/staff-umbrella/DIMA/Enzymix/logs/final/checkpoints/final.pt
echo "Generated 500 sequences"

