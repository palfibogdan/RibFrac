#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=create_env.out
#SBATCH --job-name=env

# Execute program located in $HOME
module load 2022
module load Anaconda3/2022.05
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
conda create -n medical python=3.8
conda env update -n medical -f environment-dev.yml
