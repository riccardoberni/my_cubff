#!/bin/bash
#SBATCH --job-name=fig7_lnn_full
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Fig. 7 "Long, no noise", measured as in the paper: entropy at the END of
# the run. Only the seeds that crossed the threshold in the first survey are
# re-run with --full-run; the others already ran to epoch 16000.
# Resumable: seeds already present in the output CSV are skipped.

source cluster/setup_env.sh

python3 -u my_scripts/transition_survey.py \
  --seeds "$(cat cluster/seeds_fig7_longnonoise_transitioned.txt)" \
  --mutation-prob 0 --fixed-shuffle --full-run \
  --out soup_inspector_results/survey_fig7_longnonoise_full.csv
