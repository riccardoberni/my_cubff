#!/bin/bash
#SBATCH --job-name=soup_search
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# --- Setup ambiente ---
source cluster/setup_env.sh

# --- Vai nella cartella giusta ---
cd $SLURM_SUBMIT_DIR

# --- Seed diverso per ogni task dell'array (1,2,3,4) ---
SEED=$SLURM_ARRAY_TASK_ID

python3 -u my_scripts/soup_inspector.py \
    --lang bff_noheads \
    --num-programs 131072 \
    --max-epochs 16000 \
    --seed $SEED \
    --confirm-epochs 30 \
    --results-dir soup_inspector_results
