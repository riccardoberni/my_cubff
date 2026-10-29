#!/bin/bash
#SBATCH --job-name=spatial_fig8_plot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 8, figures only: draw images/spatial_panels.pdf and
# images/fig8_ours.pdf again from the snapshots already saved by
# run_spatial_fig8.sbatch (experimentlogs/fig8_seed<N>/snapshots), without
# running the simulation again.

source cluster/setup_env.sh
mkdir -p images

DIR=$(ls -d experimentlogs/fig8_seed* | head -1)
SEED=${DIR##*seed}
echo "seed $SEED, snapshots in $DIR"

python3 my_scripts/plot_spatial_snapshots.py auto \
    --snapshot-dir $DIR/snapshots --seed $SEED \
    --out-panels images/spatial_panels.pdf --out-fig8 images/fig8_ours.pdf
