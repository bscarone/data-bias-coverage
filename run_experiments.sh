#!/bin/bash
set -e

# Defaults
EXPERIMENTS="vary_epsilon"
OBJECTIVES="min_size"
PLOTS=""
ML_EVAL=true

# Usage: ./run_experiments.sh [-e experiments] [-o objectives] [-p] [-s]
#
# Options:
#   -e  Space-separated list of experiments to run (default: "vary_epsilon")
#       Available experiments: vary_epsilon, vary_coverage, vary_cost_ratio,
#                              vary_budget, compare_strategies
#   -o  Space-separated list of ILP objectives (default: "min_size")
#       Available objectives: min_size, min_changes, min_additions
#   -p  Also generate PDF figures (default: CSVs only)
#   -s  Skip ML evaluation (step 2)
#
# Examples:
#   ./run_experiments.sh -o "min_size min_additions"
#   ./run_experiments.sh -e "vary_epsilon vary_coverage" -o "min_size min_changes min_additions"
#   ./run_experiments.sh -e "vary_epsilon vary_budget vary_cost_ratio"
#   ./run_experiments.sh -e compare_strategies -p
#   ./run_experiments.sh -e vary_epsilon -p
#   ./run_experiments.sh -e vary_epsilon -s        # solutions only, no ML eval
#   ./run_experiments.sh -e vary_epsilon -p -s     # solutions + plots, no ML eval

# Parse flags
while getopts "e:o:ps" opt; do
    case $opt in
        e) EXPERIMENTS="$OPTARG" ;;
        o) OBJECTIVES="$OPTARG" ;;
        p) PLOTS="--plots" ;;
        s) ML_EVAL=false ;;
        *) echo "Usage: $0 [-e experiments] [-o objectives] [-p] [-s]"; exit 1 ;;
    esac
done

echo "Experiments: $EXPERIMENTS"
echo "Objectives: $OBJECTIVES"
[ -n "$PLOTS" ] && echo "Plots: enabled" || echo "Plots: off (CSVs only)"
$ML_EVAL && echo "ML eval: enabled" || echo "ML eval: skipped"

# Step 1: Generate solutions for each experiment
for exp in $EXPERIMENTS; do
    echo "=== Running run_${exp}.py ==="
    python "run_${exp}.py" $PLOTS -o $OBJECTIVES
done

# Step 2: ML evaluation (only for the same experiments)
if $ML_EVAL; then
    echo "=== Running ML evaluation ==="
    python run_ml_eval.py -e $EXPERIMENTS -o $OBJECTIVES
fi