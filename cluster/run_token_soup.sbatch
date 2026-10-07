#!/bin/bash
#SBATCH --job-name=token_soup
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

# Tracer-token statistics (paper Fig. 1) for the case-study run (seed 171)
# and for a run without state transition (seed 1).

source cluster/setup_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python3 -u my_scripts/token_soup.py --seed 171 --max-epochs 5000
python3 -u my_scripts/token_soup.py --seed 1 --max-epochs 5000
