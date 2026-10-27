#!/bin/bash
#SBATCH --job-name=spatial_pilot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

# Pilot runs of the spatial soup, to measure the GPU time of each kind of run
# before choosing the number of seeds and epochs of the real experiments.
# Results go to experimentlogs/ (not versioned): they are not for the thesis.
#  A. unseeded 240 x 135 grid (paper's configuration), 16,000 epochs;
#  B. paper_fig4 replicator at the centre of the 240 x 135 grid, until it
#     occupies 90% of the grid or dies;
#  C. the same on the largest grid of the spread-speed test (256 x 256) and
#     on a well-mixed soup with the same number of programs.

source cluster/setup_env.sh

OUT=experimentlogs/spatial_pilot
mkdir -p $OUT

echo "=== A: unseeded 240 x 135, 16000 epochs ==="
python3 -u my_scripts/spatial_survey.py --seeds 1-4 --max-epochs 16000 \
    --snapshot-interval 1024 --out-dir $OUT/A_unseeded_240x135

echo "=== B: seeded centre 240 x 135 ==="
python3 -u my_scripts/spatial_survey.py --seeds 1-10 --max-epochs 4000 \
    --callback-interval 16 --snapshot-interval 128 \
    --initial-program paper_fig4 --initial-position center \
    --stop-fraction 0.9 --stop-if-extinct-after 257 \
    --out-dir $OUT/B_seeded_240x135

echo "=== C: seeded centre 256 x 256 and well-mixed 65536 ==="
python3 -u my_scripts/spatial_survey.py --seeds 1-5 --width 256 --height 256 \
    --max-epochs 6000 --callback-interval 8 --snapshot-interval 0 \
    --initial-program paper_fig4 --initial-position center \
    --stop-fraction 0.9 --stop-if-extinct-after 257 \
    --out-dir $OUT/C_seeded_256x256
python3 -u my_scripts/spatial_survey.py --seeds 1-5 --width 256 --height 256 \
    --well-mixed --max-epochs 6000 --callback-interval 8 --snapshot-interval 0 \
    --initial-program paper_fig4 \
    --stop-fraction 0.9 --stop-if-extinct-after 257 \
    --out-dir $OUT/C_seeded_wellmixed_65536

for d in $OUT/*/; do
    echo "=== summary $d ==="
    python3 my_scripts/spatial_survey.py --summarize $d/summary.csv
    python3 - "$d/summary.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    print(f"seed {r['seed']}: last_epoch {r['last_epoch']}, "
          f"extinct {r['extinct']}, {r['wall_seconds']} s")
EOF
done
