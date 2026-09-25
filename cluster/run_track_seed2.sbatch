#!/bin/bash
#SBATCH --job-name=soup_track_seed2
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_CLPhys
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source cluster/setup_env.sh
cd $SLURM_SUBMIT_DIR

python3 -u my_scripts/reseed_and_track.py \
    --from-checkpoint soup_inspector_results/checkpoints/checkpoint_20260923_175004_374310_seed2_epoch11582_score5.dat \
    --epochs 2000 \
    --print-interval 1 \
    --stop-on-extinction \
    --save-plot soup_inspector_results/seed2_track_plot.png
