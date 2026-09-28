#!/bin/bash
#SBATCH --job-name=mut_sweep
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source cluster/setup_env.sh
cd $SLURM_SUBMIT_DIR

RATES=(0 0.00012 0.00048 0.001 0.005 0.01)
NAMES=(mut0 mut012 mut048 mut1 mut5 mut10)

RATE=${RATES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

python3 -u my_scripts/transition_survey.py \
    --seeds 1-100 \
    --mutation-prob $RATE \
    --out soup_inspector_results/survey_${NAME}.csv

