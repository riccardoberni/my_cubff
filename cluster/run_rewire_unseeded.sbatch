#!/bin/bash
#SBATCH --job-name=rewire_unseeded
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 10, unseeded runs: 240 x 135 grid with random initialization,
# 16,000 epochs, seeds 1-100, for ONE fraction p of long-range interactions,
# given as argument:  sbatch cluster/run_rewire_unseeded.sbatch 0.1
# (p = 0 is Experiment 8, spatial_results/unseeded_240x135).
# Long-range interactions are obtained by rewiring the list of allowed
# interactions (spatial_survey.py --rewire-prob); the engine is not modified.
# Resumable: if the job reaches the time limit, submit it again with the
# same p.

source cluster/setup_env.sh

P=$1
if [ -z "$P" ]; then
    echo "usage: sbatch cluster/run_rewire_unseeded.sbatch <p>"
    exit 1
fi

python3 -u my_scripts/spatial_survey.py --seeds 1-100 --max-epochs 16000 \
    --snapshot-interval 0 --rewire-prob $P \
    --out-dir spatial_results/rewire_unseeded_p$P

python3 my_scripts/spatial_survey.py --summarize \
    spatial_results/rewire_unseeded_p$P/summary.csv
