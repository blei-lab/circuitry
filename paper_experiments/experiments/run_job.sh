#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --job-name="hypo-interp"
#SBATCH --output=logs/job-%j.out
#SBATCH --gres=gpu:1

SAVE_DIR="./paper_results"

python3 main.py --save_dir $SAVE_DIR --device "cuda" --task_name $1 --test_name $2 --seed $3
