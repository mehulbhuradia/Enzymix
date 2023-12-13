# Enzymix
#file transfer
scp -r processed_500 daic:~/Enzymix/

#setup
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda create -n Enzymix

conda activate Enzymix

<!-- all the required packages -->
pip3 install tqdm easydict tensorboard torch_geometric --no-cache-dir

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install packaging

