#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=sctm
#SBATCH --qos=medium         
#SBATCH --time=36:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

export HOME=/tmp/mbhuradia/

rm -rf sctm_scores_hist_by_len.pdf
rm -rf sctm_scores_hist.pdf
rm -rf sctm_scores.json

srun python bin/tmscore_training.py -p generated/pdb/
