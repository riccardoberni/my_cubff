#!/bin/bash
#SBATCH --job-name=spatial_spread
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 8, spread speed: the paper's Fig. 4 replicator placed at the
# centre of square grids of side 32, 64, 128 and 256, and in well-mixed soups
# with the same number of programs; seeds 1-200 for each configuration.
# A run stops when 90% of the programs are self-replicators, or as soon as
# the seeded replicator has died (no self-replicator at a measurement made at
# or after epoch 257). Measurements every 8 epochs.
# Resumable: if the job reaches the time limit, submit it again and the
# seeds already in each summary.csv are skipped.

source cluster/setup_env.sh

COMMON="--seeds 1-200 --max-epochs 6000 --callback-interval 8 \
    --snapshot-interval 0 --initial-program paper_fig4 \
    --stop-fraction 0.9 --stop-if-extinct-after 257"

for L in 32 64 128 256; do
    python3 -u my_scripts/spatial_survey.py $COMMON --width $L --height $L \
        --initial-position center --out-dir spatial_results/spread_grid_${L}x${L}
    python3 -u my_scripts/spatial_survey.py $COMMON --width $L --height $L \
        --well-mixed --out-dir spatial_results/spread_wellmixed_$((L * L))
done

for d in spatial_results/spread_*/; do
    echo "=== $d ==="
    python3 my_scripts/spatial_survey.py --summarize $d/summary.csv
done
