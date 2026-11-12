#!/bin/bash
#SBATCH --job-name=methods_figs
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Example figures of the terminology section of Chapter 4 (Methods):
#  - images/bff_example.pdf: step-by-step execution of a 16-cell example
#    tape, checked against the engine after every step;
#  - images/soup_snapshot.pdf: eight programs of the case-study run
#    (seed 171) at epochs 0, 2001, 2497 and 4001, with their
#    self-replication scores. The data are saved in
#    soup_inspector_results/snapshot/seed171_snapshot.npz;
#  - images/program_tape.pdf: a self-replicator of epoch 4001 and a
#    program of epoch 0 above their tape;
#  - images/longtape_excerpt.pdf: an excerpt of the long tape of seed 3
#    (longtape_results/seed3/tape_dumps_seed3);
#  - images/interaction_example.pdf: the result of executing that tape.
# The printed output of each script is saved in soup_inspector_results/snapshot/.

source cluster/setup_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p soup_inspector_results/snapshot images

python3 -u my_scripts/plot_bff_example.py --check-engine \
    --out images/bff_example.pdf \
    | tee soup_inspector_results/snapshot/bff_example.log

python3 -u my_scripts/soup_snapshot.py --seed 171 \
    --epochs 0,2001,2497,4001 \
    --data soup_inspector_results/snapshot/seed171_snapshot.npz \
    --out images/soup_snapshot.pdf \
    | tee soup_inspector_results/snapshot/seed171_snapshot.log

python3 -u my_scripts/plot_program_tape.py \
    --data soup_inspector_results/snapshot/seed171_snapshot.npz \
    --out-tape images/program_tape.pdf \
    --out-longtape images/longtape_excerpt.pdf \
    --out-interaction images/interaction_example.pdf \
    | tee soup_inspector_results/snapshot/program_tape.log
