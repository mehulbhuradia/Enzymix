# make a .sbatch file with the following content:

total = 350000
chunk_size = 10000
chunks = total // chunk_size
for i in range(0, chunks):
    with open('downloader_'+str(i)+'.sbatch', 'w') as file:
        file.write('#!/bin/sh\n')
        file.write('#SBATCH --partition=general,insy\n')
        file.write('#SBATCH --job-name=Downloader\n')
        file.write('#SBATCH --qos=long\n')
        file.write('#SBATCH --time=168:00:00\n')
        file.write('#SBATCH --ntasks=1\n')
        file.write('#SBATCH --cpus-per-task=2\n')
        file.write('#SBATCH --mem-per-cpu=16G\n')
        file.write('#SBATCH --mail-type=END\n')
        file.write('\n')
        file.write('module use /opt/insy/modulefiles\n')
        file.write('module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10\n')
        file.write('\n')
        file.write('conda activate /tudelft.net/staff-umbrella/Enzymix/Enzymix/thesis\n')
        file.write('\n')
        file.write('srun python data/download_from_af.py --start '+str(i*chunk_size)+' --chunk '+str(chunk_size)+'\n')
