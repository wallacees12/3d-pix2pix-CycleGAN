#!/usr/bin/bash -l
#SBATCH --job-name=inspect_model
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/inspect_%j.out


# Change to working directory
cd /home/sawall/3D

# Load modules
module load anaconda3
source activate sCT

# Fix MKL issues (if needed)
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

echo "🔍 Inspecting model architecture..."
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Run the inspection script
python inspect_model.py /home/sawall/scratch/checkpoints/latent_to_latent/netG_latest.pth

echo "✅ Model inspection completed!"
