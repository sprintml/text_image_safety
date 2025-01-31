#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --ntasks 16
# SBATCH --nodelist=sprint3
# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate sd3.5

python collect_activations.py --timesteps=30