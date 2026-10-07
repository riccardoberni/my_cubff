#!/bin/bash
#SBATCH --job-name=fig7_seeded
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# Fig. 7 "seeded": random soup plus the paper's hand-written self-replicator
# (the one of its Fig. 4) in a random position, run for 128 epochs, 1000
# seeds as in the paper.
# Resumable: seeds already present in the output CSV are skipped.

source cluster/setup_env.sh

python3 -u my_scripts/transition_survey.py --seeds 1-1000 --max-epochs 128 \
  --full-run --initial-program paper_fig4 \
  --out soup_inspector_results/survey_fig7_seeded_random.csv
