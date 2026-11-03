#!/bin/bash
#SBATCH --job-name=fig6_plot
#SBATCH --partition=g100_usr_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

# Experiment 4: comparison with the paper's Fig. 6 (images/fig6_replication.png)
# and rows of Table 5.2 (images/fig6_table_rows.tex), from the surveys of
# run_mutation_sweep.sbatch and run_survey.sbatch (default rate).

source cluster/setup_env.sh

python3 my_scripts/plot_fig6.py \
    soup_inspector_results/survey_mut0.csv:0.0 \
    soup_inspector_results/survey_mut012.csv:0.00012 \
    soup_inspector_results/survey_200.csv:0.00024 \
    soup_inspector_results/survey_mut048.csv:0.00048 \
    soup_inspector_results/survey_mut1.csv:0.001 \
    soup_inspector_results/survey_mut5.csv:0.005 \
    soup_inspector_results/survey_mut10.csv:0.01 \
    --out images/fig6_replication.png \
    --tex images/fig6_table_rows.tex
