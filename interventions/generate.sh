#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --ntasks 16
# SBATCH --nodelist=sprint1, sprint2
# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate sd3.5

python generate.py \
    --interventions_dir outputs_single_words/guidance_disabled_seed_42/cross_attention/timesteps_25/det0_all/interventions_global_max/25_timesteps/SD3.5 \
    --toxic_file toxic.txt \
    --non_toxic_file non_toxic.txt \
    --toxic_output_dir default_sd3/toxic \
    --non_toxic_output_dir default_sd3/non_toxic \
    --batch_size 4 \
    --timesteps 30 \
    --block_index 5