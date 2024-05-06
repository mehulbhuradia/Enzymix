# make a .sbatch file with the following content:
import os 
os.rmdir('./omegafold_predictions')
def run_sbatch(script_path):
    command = f'sbatch {script_path}'
    try:
        os.system(command)
        os.remove(script_path)
        print("sbatch command executed successfully.")
    except Exception as e:
        print(f"Error running sbatch command: {e} {command}")


allfiles = os.listdir('./proteinmpnn_residues')
total = len(allfiles)
fullpaths = [os.path.join('./proteinmpnn_residues', f) for f in allfiles]
chunk_size = 4
chunks = total // chunk_size
chunks = 2

for i in range(0, chunks):
    with open('om'+str(i)+'.sbatch', 'w') as file:
        file.write('#!/bin/sh\n')
        file.write('#SBATCH --partition=general,insy\n')
        file.write('#SBATCH --job-name=omp\n')
        file.write('#SBATCH --qos=short\n') #short works for 500 structuresas it only takes like 10ish mins
        file.write('#SBATCH --time=4:00:00\n')
        file.write('#SBATCH --ntasks=1\n')
        file.write('#SBATCH --cpus-per-task=2\n')
        file.write('#SBATCH --mem-per-cpu=32G\n')
        file.write('#SBATCH --mail-type=END\n')
        file.write('#SBATCH --gres=gpu:a40\n')
        file.write('\n')
        file.write('module use /opt/insy/modulefiles\n')
        file.write('module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10\n')
        file.write('\n')
        file.write('conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis\n')
        file.write('\n')
        file.write('export HOME=/tmp/mbhuradia/')
        file.write('\n')
        fastas = ' '.join(fullpaths[i*chunk_size:(i+1)*chunk_size])
        file.write('srun python bin/omegafold_across_gpus.py '+fastas+' --start '+str(i*chunk_size)+'\n')
    run_sbatch('om'+str(i)+'.sbatch')
