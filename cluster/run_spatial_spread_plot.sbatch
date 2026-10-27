#!/bin/bash
#SBATCH --job-name=spatial_spread_plot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 8, spread speed: table, fits and figure from the results of
# run_spatial_spread.sbatch (spatial_results/spread_*/summary.csv).

source cluster/setup_env.sh

python3 my_scripts/plot_spatial_spread.py --results-dir spatial_results \
    --out images/spatial_spread.pdf
