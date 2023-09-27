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
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

cd RibFrac
srun python Code/u-net.py --mode train --train_folder "data/train" --val_folder "data/val"
