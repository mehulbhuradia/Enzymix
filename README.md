# Enzymix
#file transfer
scp -r processed_500 daic:~/Enzymix/


<!-- diffab data -->
scp -r all_structures daic:/tudelft.net/staff-umbrella/Enzymix/diffab/data/


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




sbatch script.sbatch name layers addlayers

eg:

sbatch script.sbatch bigacomplex 2 4


squeue -u mbhuradia

squeue -u mbhuradia -t RUNNING -h | wc -l

14 x 4 =206078 egcl layers ran out of memory

8 x 8 = 188888 ran out

RuntimeError: DataLoader worker (pid 191071) is killed by signal: Killed.
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=9209586.0. Some of your processes may have been killed by the cgroup out-$
srun: error: influ2: task 0: Out Of Memory
srun: launch/slurm: _step_signal: Terminating StepId=9209586.0
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=9209586.batch. Some of your processes may have been killed by the cgroup $


# print("pout",p_pred[0,0])
        # print("pin",p_noisy[0,0])
        #  todo figure out hy its the same?
didnt work cuz x was being changed



Increasing lr to 2, 4 and 8 is always bad regardless of change in batch size
using a large number(16) of layers causes nans
Init weight made the loss explode for 8 layers, should i try weight decay?