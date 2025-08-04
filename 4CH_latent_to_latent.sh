#!/usr/bin/bash -l
#SBATCH --job-name=4CH_L2L
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=/home/sawall/3D/logs_latent_to_latent_%j.out

module load anaconda3
source activate sCT

echo "Running on host : $(hostname)"
echo "Starting 3D Latent-to-Latent Pix2Pix Training"
nvidia-smi

cd /home/sawall/3D

# Run latent-to-latent training
python train_4channel_256.py \
    --dataroot /home/sawall/scratch/latent_data/latent_pairs/ \
    --checkpoints_dir /home/sawall/scratch/checkpoints \
    --which_model_netG unet_4channel_256 \
    --name Latent_to_Latent_256 \
    --print_freq 10 \
    --save_epoch_freq 20 \
    --model pix2pix3d \
    --input_nc 4 \
    --output_nc 4 \
    --depthSize 64 \
    --fineSize 256 \
    --batchSize 2 \
    --niter 200 \
    --niter_decay 800 \
    --lr 0.0001 \
    --lambda_A 50.0
