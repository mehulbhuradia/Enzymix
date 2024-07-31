#!/bin/sh
#SBATCH --partition=general,insy
#SBATCH --job-name=analysis
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
srun python esm_perp.py --in_fasta generated/fasta/generated.fasta
echo "ESM Perplexity"
srun python bin/omegafold_across_gpus.py generated/fasta/generated.fasta -o generated/fasta/omega
echo "OmegaFold"
srun python b_factor.py --in_folder generated/fasta/omega
echo "B-Factor"
srun python bin/pdb_to_residue_proteinmpnn.py generated/pdb/
echo "ProteinMPNN Residues"
srun python bin/omegafold_across_gpus.py proteinmpnn_residues/*.fasta
echo "OmegaFold for ProteinMPNN Residues"
srun python bin/sctm.py -p generated/pdb/
echo "SCTM"
