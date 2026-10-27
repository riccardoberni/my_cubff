#!/bin/bash
#SBATCH --job-name=spatial_unseeded
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 8: spatial soup of the paper (240 x 135 grid, radius 2, default
# mutation rate), random initialization, 16,000 epochs, seeds 1-100.
# No snapshots: the runs are deterministic, so the seeds with a transition
# are run again later with frequent snapshots for the figures.
# Resumable: if the job reaches the time limit, submit it again and the
# seeds already in summary.csv are skipped.

source cluster/setup_env.sh

python3 -u my_scripts/spatial_survey.py --seeds 1-100 --max-epochs 16000 \
    --snapshot-interval 0 --out-dir spatial_results/unseeded_240x135

python3 my_scripts/spatial_survey.py --summarize \
    spatial_results/unseeded_240x135/summary.csv
