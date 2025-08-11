#!/bin/bash
#SBATCH --job-name=model_inspect
#SBATCH --output=logs/model_inspect_%j.out
#SBATCH --error=logs/model_inspect_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00

echo "🚀 Starting model inspection on GPU node"
module load anaconda3
source activate sCT

python batch_model_inspect.py


