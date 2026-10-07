#!/bin/bash
#SBATCH --job-name=long_full
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Fig. 7 "Long" and Fig. 5, seeds 101-200, run to 16000 epochs so that the
# end-of-run entropy and the full entropy trace are available for every seed.
# Resumable: seeds already present in the output CSV are skipped.

source cluster/setup_env.sh

python3 -u my_scripts/transition_survey.py --seeds 101-200 --full-run \
  --trace-dir soup_inspector_results/traces_fig5 \
  --out soup_inspector_results/survey_fig5_trace_index.csv
