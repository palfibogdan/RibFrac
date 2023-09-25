#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --output=download_data.out
#SBATCH --job-name=data


# Execute program located in $HOME
conda activate medical
cd RibFrac
srun python code/download_data.py