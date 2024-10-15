#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --job-name="hypo-interp"
#SBATCH --output=logs/job-%j.out
#SBATCH --gres=gpu:1

SAVE_DIR="./paper_results"

python3 main.py --save_dir $SAVE_DIR --device "cuda" --task_name $1 --test_name $2 --seed $3 --sweep --param_name test.size_random_circuits --sweep_values 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
