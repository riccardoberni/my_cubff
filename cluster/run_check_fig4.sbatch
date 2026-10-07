#!/bin/bash
#SBATCH --job-name=check_fig4
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

# Cross-check of plot_fig4.py against the engine (bin/main --run) for the
# 23 steps shown in the paper's Fig. 4.

source cluster/setup_env.sh

python3 -u my_scripts/check_fig4_engine.py --main bin/main
