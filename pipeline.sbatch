#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=Pipeline
#SBATCH --qos=long         
#SBATCH --time=168:00:00      
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

srun python generate500.py --resume /tudelft.net/staff-umbrella/DIMA/Enzymix/logs/final/checkpoints/final.pt
srun python esm_perp.py --in_fasta generated/fasta/generated.fasta
srun python bin/omegafold_across_gpus.py generated/fasta/generated.fasta -o generated/fasta/omega
srun python b_factor.py --in_folder generated/fasta/omega
srun python bin/pdb_to_residue_proteinmpnn.py generated/pdb/
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/1*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/2*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/3*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/4*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/5*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/6*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/7*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/8*.fasta
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/9*.fasta
srun python bin/sctm.py -p generated/pdb/
