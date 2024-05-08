#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=F_Enzymix
#SBATCH --qos=medium         
#SBATCH --time=36:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

srun python bin/pdb_to_residue_proteinmpnn.py generated/pdb/
