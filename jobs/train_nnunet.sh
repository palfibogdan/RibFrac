#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --output=train_nnunet.out
#SBATCH --job-name=train_nnunet

# Execute program located in $HOME
source activate medical

cd RibFrac
srun python Code/nn-unet.py --mode train --train_folder "data/train" --val_folder "data/val"