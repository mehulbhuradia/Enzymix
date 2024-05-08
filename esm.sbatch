#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=esm
#SBATCH --qos=short         
#SBATCH --time=4:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

export HF_HOME=/tmp/mbhuradia/

srun python esm_perp.py --in_fasta generated/fasta/generated.fasta