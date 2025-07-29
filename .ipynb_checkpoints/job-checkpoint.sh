#!/usr/bin/bash -l
#SBATCH --job-name=test_pipeline
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

./run_full_pipeline.sh