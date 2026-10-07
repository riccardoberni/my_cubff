#!/bin/bash
#SBATCH --job-name=fig5_fig7
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out

source cluster/setup_env.sh

case $SLURM_ARRAY_TASK_ID in
  0)
    # Fig. 7 "Short": 128 epoche, unseeded, mutation rate di default
    python3 -u my_scripts/transition_survey.py --seeds 1-100 --max-epochs 128 \
      --out soup_inspector_results/survey_fig7_short.csv
    ;;
  1)
    # Fig. 7 "Long, no noise": 16000 epoche, mutazione zero, shuffle deterministico
    python3 -u my_scripts/transition_survey.py --seeds 1-100 --mutation-prob 0 --fixed-shuffle \
      --out soup_inspector_results/survey_fig7_longnonoise.csv
    ;;
  2)
    # Fig. 5: traccia completa dell'entropia epoca per epoca, mutation rate di default
    python3 -u my_scripts/transition_survey.py --seeds 1-100 --full-run \
      --trace-dir soup_inspector_results/traces_fig5 \
      --out soup_inspector_results/survey_fig5_trace_index.csv
    ;;
esac