#!/bin/bash
#SBATCH --job-name=survey
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source cluster/setup_env.sh
cd $SLURM_SUBMIT_DIR

python3 -u my_scripts/transition_survey.py \
    --seeds 1-200 \
    --out soup_inspector_results/survey_200.csv
