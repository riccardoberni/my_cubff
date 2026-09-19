#!/bin/bash
#SBATCH --job-name=soup_inspector
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Environment setup ---
source cluster/setup_env.sh

# --- Move to the submit directory ---
cd $SLURM_SUBMIT_DIR

# --- Run the script ---
python3 -u my_scripts/soup_inspector.py \
    --lang bff_noheads \
    --num-programs 131072 \
    --max-epochs 16000 \
    --seed 128 \
    --phase search \
    --checkpoint checkpoint.dat \
    --confirm-epochs 5