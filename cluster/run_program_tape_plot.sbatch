#!/bin/bash
#SBATCH --job-name=program_tape
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

# Redraws images/program_tape.pdf (programs and tape of the primordial
# soup), images/longtape_excerpt.pdf (an excerpt of the long tape of seed 3)
# and images/interaction_example.pdf, from the data already saved by
# cluster/run_methods_figures.sbatch. The self-replication scores are
# computed by the engine, which needs the GPU.

source cluster/setup_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python3 -u my_scripts/plot_program_tape.py \
    --data soup_inspector_results/snapshot/seed171_snapshot.npz \
    --longtape-dump longtape_results/seed3/tape_dumps_seed3/tape_000050002267.bin \
    --longtape-start 21312 \
    --out-tape images/program_tape.pdf \
    --out-longtape images/longtape_excerpt.pdf \
    --out-interaction images/interaction_example.pdf \
    | tee soup_inspector_results/snapshot/program_tape.log
