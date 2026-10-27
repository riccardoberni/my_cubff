#!/bin/bash
#SBATCH --job-name=spatial_wellmixed
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Control for Experiment 8: well-mixed soup (standard random pairing) with the
# same number of programs as the 240 x 135 grid (32,400), random
# initialization, 16,000 epochs, seeds 1-100. Separates the effect of space
# from the effect of the smaller population.
# Resumable: if the job reaches the time limit, submit it again and the
# seeds already in summary.csv are skipped.

source cluster/setup_env.sh

python3 -u my_scripts/spatial_survey.py --seeds 1-100 --max-epochs 16000 \
    --well-mixed --width 240 --height 135 \
    --snapshot-interval 0 --out-dir spatial_results/wellmixed_32400

python3 my_scripts/spatial_survey.py --summarize \
    spatial_results/wellmixed_32400/summary.csv
