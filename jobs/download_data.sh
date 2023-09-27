#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --output=download_data.out
#SBATCH --job-name=data


# Run script from $HOME
cd RibFrac
srun python Code/download_data.py