#!/usr/bin/bash -l
#SBATCH --job-name=4CH
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=/home/sawall/3D/logs_batch_4.out

module load anaconda3
source activate sCT

echo "Running on host : $(hostname)"
echo "Starting 3DPix2Pix Overfit on 1ABA001"
nvidia-smi

cd /home/sawall/3D

# Run training

python train_4channel_256.py \
    --dataroot /home/sawall/scratch/latent_data/latent_scaled/HN/ \
    --checkpoints_dir /home/sawall/scratch/checkpoints \
    --which_model_netG unet_4channel_256 \
    --name HN_256 \
    --print_freq 10 \
    --save_epoch_freq 20 \
    --model pix2pix3d \
    --input_nc 4 \
    --output_nc 1 \
    --depthSize 64 \
    --fineSize 256 \
    --batchSize 8 \
    --niter 100 \
    --niter_decay 1000
