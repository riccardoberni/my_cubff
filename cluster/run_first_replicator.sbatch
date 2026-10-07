#!/bin/bash
#SBATCH --job-name=first_rep
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Tracer tokens: find the epoch and the pair of programs where the first
# self-replicator of the case-study run (seed 171) emerges.

source cluster/setup_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python3 -u my_scripts/find_first_replicator.py --seed 171
