#!/bin/bash
#SBATCH --job-name=rewire_seeded
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 10, seeded runs: the paper's Fig. 4 replicator at the centre of
# the 240 x 135 grid, for each fraction p of long-range interactions; seeds
# 1-100 for each p. Long-range interactions are obtained by rewiring the list
# of allowed interactions given to the engine (spatial_survey.py
# --rewire-prob): the engine itself is not modified.
# A run stops when 90% of the programs are self-replicators or when the
# seeded replicator has died; the grid at the end of every surviving run is
# saved for the spatial measures.
# Resumable: if the job reaches the time limit, submit it again.

source cluster/setup_env.sh

for P in 0 0.001 0.01 0.05 0.1 0.25 0.5 1; do
    python3 -u my_scripts/spatial_survey.py --seeds 1-100 --max-epochs 6000 \
        --callback-interval 16 --snapshot-interval 0 --snapshot-final \
        --initial-program paper_fig4 --initial-position center \
        --stop-fraction 0.9 --stop-if-extinct-after 257 \
        --rewire-prob $P --out-dir spatial_results/rewire_seeded_p$P
done
