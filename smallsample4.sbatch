#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=sampling
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

rm -rf proteinmpnn_residues
rm -rf generated
rm -rf omegafold_predictions
rm -rf sctm_scores_hist_by_len.pdf
rm -rf sctm_scores_hist.pdf
rm -rf sctm_scores.json

srun python sample.py --resume ./logs/4_cont/checkpoints/650.pt --num_samples 5000 --batch_size 10 --output generated4/ --layers 4
srun python esm_perp.py --in_fasta generated4/fasta/generated.fasta
srun python bin/omegafold_across_gpus.py generated4/fasta/generated.fasta -o generated4/fasta/omega
srun python b_factor.py --in_folder generated4/fasta/omega