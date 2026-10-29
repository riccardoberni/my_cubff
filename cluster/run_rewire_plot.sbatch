#!/bin/bash
#SBATCH --job-name=rewire_plot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 10: table and figure (images/rewire.pdf) from the runs of
# run_rewire_seeded.sbatch and run_rewire_unseeded.sbatch.

source cluster/setup_env.sh

python3 my_scripts/plot_rewire.py --results-dir spatial_results \
    --out images/rewire.pdf
