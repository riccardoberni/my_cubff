#!/bin/bash
#SBATCH --job-name=longtape_plot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Experiments 6 and 7, analysis only: figure and table of Experiment 6
# (images/longtape_seed0.pdf) and analysis of Experiment 7
# (longtape_results/patterns/) from the runs of run_longtape.sbatch, without
# running the simulations again.

source cluster/setup_env.sh

mkdir -p images
python3 my_scripts/plot_longtape.py \
    --jsonl longtape_results/seed0/seed0_2000000000instr.jsonl \
            longtape_results/seed1/seed1_2000000000instr.jsonl \
            longtape_results/seed2/seed2_2000000000instr.jsonl \
            longtape_results/seed3/seed3_2000000000instr.jsonl \
            longtape_results/seed4/seed4_2000000000instr.jsonl \
            longtape_results/seed5/seed5_2000000000instr.jsonl \
    --out images/longtape_seed0.pdf
python3 -u my_scripts/longtape_patterns.py --results-dir longtape_results \
    --out-dir longtape_results/patterns
