#!/bin/bash
#SBATCH --job-name=soup_track
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:45:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source cluster/setup_env.sh
cd $SLURM_SUBMIT_DIR

CHECKPOINTS=(
    "soup_inspector_results/checkpoints/checkpoint_20260924_180819_104801_seed2_epoch11862_score64.dat"
    "soup_inspector_results/checkpoints/checkpoint_20260924_140036_279794_seed7_epoch15288_score64.dat"
    "soup_inspector_results/checkpoints/checkpoint_20260924_140548_473380_seed8_epoch3593_score64.dat"
    "soup_inspector_results/checkpoints/checkpoint_20260924_155255_256488_seed11_epoch15353_score22.dat"
)
SEEDS=(2 7 8 11)

CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python3 -u my_scripts/reseed_and_track.py \
    --from-checkpoint "$CKPT" \
    --seed $SEED \
    --epochs 3000 \
    --print-interval 1 \
    --stop-on-extinction \
    --full-selfrep \
    --save-metrics soup_inspector_results/track_seed${SEED}_metrics.jsonl \
    --save-plot soup_inspector_results/track_seed${SEED}_plot.png
