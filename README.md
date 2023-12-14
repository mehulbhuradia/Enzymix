# Enzymix
#file transfer
scp -r processed_500 daic:~/Enzymix/

#setup
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda create -n Enzymix

conda create --prefix ./thesis

conda activate Enzymix

conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis

cd /tudelft.net/staff-umbrella/Enzymix/Enzymix/

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

