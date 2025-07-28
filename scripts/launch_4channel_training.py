#!/usr/bin/env python3
"""
Launch Script for 4-Channel Training
Easy-to-use script to start 4-channel MR→CT training

Usage:
    python launch_4channel_training.py
    python launch_4channel_training.py --gpu_ids 0,1  # Multi-GPU
    python launch_4channel_training.py --help         # See all options
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch 4-Channel Training')
    
    # Essential arguments
    parser.add_argument('--name', type=str, default='4channel_mr_to_ct', 
                       help='experiment name')
    parser.add_argument('--dataroot', type=str, default='./datasets/all_channels',
                       help='path to 4-channel dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', 
                       help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='initial learning rate')
    parser.add_argument('--niter', type=int, default=100,
                       help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                       help='# of iter to linearly decay learning rate to zero')
    
    # Architecture choice
    parser.add_argument('--architecture', type=str, default='unet_4channel_128',
                       choices=['unet_4channel_128', 'unet_4channel_256',
                               'resnet_4channel_6blocks', 'resnet_4channel_9blocks',
                               'dense_4channel'],
                       help='4-channel generator architecture')
    
    # Training options
    parser.add_argument('--augment', action='store_true',
                       help='enable data augmentation')
    parser.add_argument('--normalize', action='store_true',
                       help='normalize input data')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='use mixed precision training')
    
    args = parser.parse_args()
    
    # Validate dataset
    if not os.path.exists(args.dataroot):
        print(f"❌ Dataset not found: {args.dataroot}")
        print("   Make sure you've run 4Channel_data.ipynb to create the dataset")
        return
    
    crops_file = os.path.join(args.dataroot, "crops.pkl")
    if not os.path.exists(crops_file):
        print(f"❌ Dataset metadata not found: {crops_file}")
        print("   Make sure the dataset was created properly")
        return
    
    print("✅ Dataset validation passed")
    
    # Build command
    cmd_parts = [
        sys.executable,  # Python executable
        "train_4channel.py",
        f"--name {args.name}",
        f"--dataroot {args.dataroot}",
        f"--gpu_ids {args.gpu_ids}",
        f"--batchSize {args.batch_size}",
        f"--lr {args.lr}",
        f"--niter {args.niter}",
        f"--niter_decay {args.niter_decay}",
        f"--which_model_netG {args.architecture}",
        "--input_nc 4",
        "--output_nc 1",
        "--lambda_L1 100.0",
        "--use_4channel_dataset",
        "--display_freq 100",
        "--print_freq 50",
        "--save_epoch_freq 10"
    ]
    
    # Add optional flags
    if args.augment:
        cmd_parts.append("--augment_data")
    
    if args.normalize:
        cmd_parts.append("--normalize_input")
    
    if args.mixed_precision:
        cmd_parts.append("--mixed_precision")
    
    # Join command
    cmd = " ".join(cmd_parts)
    
    print("🚀 Starting 4-Channel Training...")
    print(f"📋 Command: {cmd}")
    print(f"📁 Dataset: {args.dataroot}")
    print(f"🏗️ Architecture: {args.architecture}")
    print(f"💾 Experiment: {args.name}")
    print(f"🎮 GPU(s): {args.gpu_ids}")
    print()
    
    # Execute training
    os.system(cmd)

if __name__ == "__main__":
    main()
