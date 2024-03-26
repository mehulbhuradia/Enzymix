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


9854113 insy,gene dima_dif mbhuradi PD       0:00      1 (Falied) L_eps og afdb Enzymix/dima (Nan or inf detected)
9854340 insy,gene dima_dif mbhuradi PD       0:00      1 (Priority) L_score og afdb Enzymix/dima (Nan or inf detected)


9853616   general dima_dif mbhuradi  R 1-02:44:08      1 gpu11 og afdb no lr decay Enzymix/dima
9854375      insy dima_dif mbhuradi  R    4:42:41      1 gpu21
9854362      insy  Enzymix mbhuradi  R    4:56:09      1 gpu01 e5
9854361      insy  Enzymix mbhuradi  R    4:57:15      1 gpu01 e4
9854334      insy  Enzymix mbhuradi  R    5:37:10      1 gpu20 e1
9854116      insy  Enzymix mbhuradi  R    8:14:06      1 insy16 e3
9854115      insy  Enzymix mbhuradi  R    8:16:06      1 insy16 e2
9853329      insy dima_dif mbhuradi  R 1-06:05:28      1 gpu21 cross DIMA/dima
9853504      insy dima_inf mbhuradi  R 1-04:49:18      1 gpu22 og afdb Enzymix/dima
9854976 insy,gene  Enzymix mbhuradi PD       0:00      1 (Priority) e6