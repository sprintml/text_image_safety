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

python aura_diffusion.py \
    --auroc-scores-dir outputs_single_words/guidance_disabled_seed_42/cross_attention/timesteps_25/auroc_stats\
    --tag 25_timesteps \
    --model_name SD3.5 \
    --intervention det0 \
    --interventions-cache-dir outputs_single_words/guidance_disabled_seed_42/cross_attention/timesteps_25/det0_all/interventions_global_max