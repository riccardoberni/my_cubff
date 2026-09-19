#!/bin/bash
#SBATCH --job-name=soup_inspector
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Setup ambiente ---
source setup_env.sh

# --- Vai nella cartella giusta ---
cd $SLURM_SUBMIT_DIR

# --- Esegui il tuo wrapper ---
python3 -u soup_inspector.py \
    --lang bff_noheads \
    --num-programs 131072 \
    --max-epochs 16000 \
    --seed 128 \
    --phase search \
    --checkpoint checkpoint.dat \
    --confirm-epochs 5