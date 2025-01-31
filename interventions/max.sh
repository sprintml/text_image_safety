#!/bin/bash
# SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH --ntasks 16
# SBATCH --nodelist=sprint1
# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate sd3.5

python global_max.py \
    --input_dir /home/aditya/diffusers/sd3_cross_attention_layer10