#!/usr/bin/env python3
"""
Launch script for 4-channel model testing
Provides an easy interface to test trained 4-channel models
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def validate_model_exists(checkpoints_dir, model_name, epoch):
    """Validate that the model checkpoint exists"""
    checkpoint_dir = Path(checkpoints_dir) / model_name
    
    # Common checkpoint naming patterns
    checkpoint_patterns = [
        f"netG_{epoch}.pth",
        f"{epoch}_net_G.pth", 
        f"netG_epoch_{epoch}.pth",
        "netG_latest.pth"
    ]
    
    for pattern in checkpoint_patterns:
        checkpoint_path = checkpoint_dir / pattern
        if checkpoint_path.exists():
            print(f"✅ Found model checkpoint: {checkpoint_path}")
            return True
    
    print(f"❌ Model checkpoint not found in {checkpoint_dir}")
    print(f"   Looked for: {checkpoint_patterns}")
    return False

def validate_dataset_exists(dataroot):
    """Validate that the dataset exists"""
    dataset_path = Path(dataroot)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataroot}")
        return False
    
    crops_file = dataset_path / "crops.pkl"
    if not crops_file.exists():
        print(f"❌ Dataset metadata not found: {crops_file}")
        return False
    
    print(f"✅ Dataset validation passed: {dataroot}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Launch 4-Channel Model Testing')
    
    # Required arguments
    parser.add_argument('--name', required=True, help='Name of the experiment (model)')
    parser.add_argument('--dataroot', required=True, help='Path to dataset directory')
    
    # Optional arguments with defaults
    parser.add_argument('--gpu_ids', default='0', help='GPU IDs (e.g., 0,1 or -1 for CPU)')
    parser.add_argument('--which_epoch', default='latest', help='Which epoch to test')
    parser.add_argument('--how_many', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--which_model_netG', default='unet_4channel_128', 
                       choices=['unet_4channel_128', 'unet_4channel_256', 'resnet_4channel_9blocks', 
                               'resnet_4channel_6blocks', 'dense_4channel'],
                       help='4-channel generator architecture')
    
    args = parser.parse_args()
    
    print("🧪 4-Channel Model Testing Launcher")
    print("=" * 50)
    
    # Validation
    print("🔍 Validating configuration...")
    
    if not validate_dataset_exists(args.dataroot):
        sys.exit(1)
    
    if not validate_model_exists(args.checkpoints_dir, args.name, args.which_epoch):
        sys.exit(1)
    
    print("✅ Validation passed")
    
    # Build command
    cmd = [
        sys.executable, 'test_4channel.py',
        '--name', args.name,
        '--dataroot', args.dataroot,
        '--gpu_ids', args.gpu_ids,
        '--which_epoch', args.which_epoch,
        '--how_many', str(args.how_many),
        '--checkpoints_dir', args.checkpoints_dir,
        '--results_dir', args.results_dir,
        '--which_model_netG', args.which_model_netG,
        '--input_nc', '4',
        '--output_nc', '1',
        '--dataset_mode', 'four_channel',
        '--no_flip',
        '--serial_batches'
    ]
    
    print(f"\n🚀 Starting 4-Channel Testing...")
    print(f"📋 Command: {' '.join(cmd)}")
    print(f"📁 Dataset: {args.dataroot}")
    print(f"🏗️ Architecture: {args.which_model_netG}")
    print(f"💾 Experiment: {args.name}")
    print(f"📅 Epoch: {args.which_epoch}")
    print(f"🎮 GPU(s): {args.gpu_ids}")
    print(f"📊 Samples: {args.how_many}")
    
    # Execute
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Testing completed successfully!")
        
        # Show output directories
        results_path = Path(args.results_dir) / args.name / f"test_{args.which_epoch}_npz"
        print(f"📁 Results saved to: {results_path}")
        print(f"🔍 Use Visualize_NPZ_Results.ipynb to inspect results")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Testing failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ Testing interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
