#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --output=train_unet.out
#SBATCH --job-name=train_unet


# Execute program located in $HOME
conda activate medical
cd RibFrac
srun python Code/train_unet.py