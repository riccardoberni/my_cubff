#!/bin/bash
#SBATCH --job-name=longtape
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Experiments 6 and 7: long-tape runs of seeds 0-5 (65,536-byte tape, head1
# offset 16, one mutation every 400,000 steps, 2e9 steps per seed, entropy
# every 5e6 steps, tape snapshots every 5e7 steps), one process per seed on
# the CPU, into longtape_results/. Then the figure and the table of
# Experiment 6 (images/longtape_seed0.pdf) and the analysis of Experiment 7
# (longtape_results/patterns/).
# Resumable: a seed whose final checkpoint exists is complete and is
# skipped; an incomplete seed is started again from scratch.

source cluster/setup_env.sh

mkdir -p longtape_results images
for S in 0 1 2 3 4 5; do
    if [ -f longtape_results/seed$S/seed${S}_2000000000instr_ckpt.dat ]; then
        echo "seed $S already complete"
        continue
    fi
    rm -rf longtape_results/seed$S
    python3 -u my_scripts/long_tape_bff.py --seed $S \
        --max-instructions 2000000000 --report-every 5000000 \
        --dump-tape-every 50000000 --output-dir longtape_results \
        > longtape_results/seed$S.log 2>&1 &
done
wait

python3 my_scripts/plot_longtape.py \
    --jsonl longtape_results/seed0/seed0_2000000000instr.jsonl \
            longtape_results/seed1/seed1_2000000000instr.jsonl \
            longtape_results/seed2/seed2_2000000000instr.jsonl \
            longtape_results/seed3/seed3_2000000000instr.jsonl \
            longtape_results/seed4/seed4_2000000000instr.jsonl \
            longtape_results/seed5/seed5_2000000000instr.jsonl \
    --out images/longtape_seed0.pdf
python3 -u my_scripts/longtape_patterns.py --results-dir longtape_results \
    --out-dir longtape_results/patterns
