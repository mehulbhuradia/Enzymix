# Enzymix
#file transfer
scp -r processed_500 daic:~/Enzymix/


<!-- diffab data -->
scp -r all_structures daic:/tudelft.net/staff-umbrella/Enzymix/diffab/data/

scp -r zipz daic:/tudelft.net/staff-umbrella/Enzymix/LatentDiff/

scp -r LatentDiff daic:/tudelft.net/staff-umbrella/Enzymix/

scp -r af_structures daic:/tudelft.net/staff-umbrella/Enzymix/se3_diffusion/

scp -r uniprot_sprot.fasta daic:/tudelft.net/staff-umbrella/Mehul/Enzymix/


#setup
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda create -n Enzymix

conda create --prefix ./thesis

conda activate Enzymix

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

cd /tudelft.net/staff-umbrella/Enzymix/Enzymix/

<!-- download log files -->
scp -r daic:/tudelft.net/staff-umbrella/Enzymix/Enzymix/logs/ ./

scp -r daic:/tudelft.net/staff-umbrella/Mehul/Enzymix/logs/ ./


scp -r daic:/tudelft.net/staff-umbrella/Enzymix/Enzymix/logs/train_2024_03_06__16_44_22gcl_long_tateverylayer_layers_2_add_layers_24 ./logs/
# tensorboard --logdir D:\Thesis\Enzymix\logs\

<!-- cuz limited space -->
conda config --add pkgs_dirs /tmp/

<!-- all the required packages -->
<!-- doesnt work in project dir  but conda install works-->
pip3 install tqdm easydict tensorboard torch_geometric --no-cache-dir

conda install -c anaconda tqdm
conda install -c conda-forge easydict
conda install -c conda-forge tensorboard
conda install pyg -c pyg
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install packaging

<!-- for esm2 -->
conda install conda-forge::transformers
<!-- for omegafold -->
conda install conda-forge::biopython
<!-- to read fasta -->
conda install bioconda::pyfaidx


<!-- to change tmpdir, very imp for omega fold -->
export HOME=/tmp/mbhuradia/



sbatch script.sbatch name layers addlayers

eg:

sbatch script.sbatch bigacomplex 2 4


squeue -u mbhuradia

squeue -u mbhuradia -t RUNNING -h | wc -l



for /L %i in (0, 1, 1000) do python visall.py --n %i

conda env create -f se3.yml --prefix ./se3

9651725: gat
slurm-9651732.out: linear 10
slurm-9651731.out: linear 10


Long jobs:
9651738 linear 20
important:9651778: gcl
9652891: gcl t long -i will use

v100 is the only gpu that worked for se3

9683011
9683010
9683009
9683008
9683007
9683006


conda activate /tudelft.net/staff-umbrella/Mehul/OmegaFold/omega