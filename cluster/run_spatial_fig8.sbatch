#!/bin/bash
#SBATCH --job-name=spatial_fig8
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 8, figure: among the runs of run_spatial_unseeded.sbatch, take
# the one whose self-replicators reach 50% of the grid earliest, run it again
# (the engine is deterministic, so it is the same run) saving the grid every
# 64 epochs from shortly before the replicators appear, and draw:
#   images/spatial_panels.pdf  the whole grid at four moments of the wave;
#   images/fig8_ours.pdf       the layout of the paper's Fig. 8.
# The epochs, the crop and the column printed as text are printed below.

source cluster/setup_env.sh
mkdir -p images

read SEED FROM <<< $(python3 - <<'PYEOF'
import csv
rows = [r for r in csv.DictReader(open("spatial_results/unseeded_240x135/summary.csv"))
        if r["epoch_selfrep_50pct"]]
r = min(rows, key=lambda r: int(r["epoch_selfrep_50pct"]))
print(r["seed"], max(1, int(r["epoch_selfrep_1pct"]) - 256))
PYEOF
)
echo "seed $SEED, snapshots from epoch $FROM"

python3 -u my_scripts/spatial_survey.py --seeds $SEED --max-epochs 16000 \
    --snapshot-interval 64 --snapshot-from $FROM --stop-fraction 0.9 \
    --out-dir experimentlogs/fig8_seed$SEED

python3 my_scripts/plot_spatial_snapshots.py auto \
    --snapshot-dir experimentlogs/fig8_seed$SEED/snapshots --seed $SEED \
    --out-panels images/spatial_panels.pdf --out-fig8 images/fig8_ours.pdf
