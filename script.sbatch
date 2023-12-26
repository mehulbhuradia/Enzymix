#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --job-name=Enzymix_$1_$2_$3
#SBATCH --qos=long         
#SBATCH --time=168:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu:turing

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python train.py train.yml --name $1 --layers $2 --add_layers $3