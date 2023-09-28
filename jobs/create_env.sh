#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=create_env.out
#SBATCH --job-name=env

# # Run script from $HOME
module purge
module load 2022
module load Anaconda3/2022.05

conda create -n medical python=3.8 -y
conda activate medical
conda install -c conda-forge monai
conda install -c pytorch ignite
conda install -c conda-forge packaging
conda init --all