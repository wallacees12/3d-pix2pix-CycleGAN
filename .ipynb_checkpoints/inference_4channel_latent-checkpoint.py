#!/usr/bin/env python3
"""
4-Channel Latent-to-Latent Inference Script
Run inference with your trained 4-channel latent-to-latent Pix2Pix model

This script supports:
- Latent-to-latent translation (4→4 channels)
- Processing individual files or entire directories
- Multiple output formats (NPZ, visualization, etc.)
- Batch processing for efficiency

Usage:
    # Process single file
    python inference_4channel_latent.py --name experiment_name --input_file path/to/latent.npz
    
    # Process directory
    python inference_4channel_latent.py --name experiment_name --input_dir path/to/latent_files/
    
    # Specify output directory and format
    python inference_4channel_latent.py --name experiment_name --input_dir data/ --output_dir results/ --save_format npz
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def parse_inference_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='4-Channel Latent-to-Latent Inference')
    
    # Model parameters
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment (checkpoint folder name)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Models are saved here')
    parser.add_argument('--which_epoch', type=str, default='latest', help='Which epoch to load (latest, best, or specific number)')
    
    # Network parameters (should match training)
    parser.add_argument('--input_nc', type=int, default=4, help='Number of input channels')
    parser.add_argument('--output_nc', type=int, default=4, help='Number of output channels')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters')
    parser.add_argument('--which_model_netG', type=str, default='unet_4channel_128', 
                       help='Generator architecture')
    parser.add_argument('--norm', type=str, default='batch', help='Normalization type')
    parser.add_argument('--no_dropout', action='store_true', help='Disable dropout')
    
    # Input/Output
    parser.add_argument('--input_file', type=str, help='Single input NPZ file to process')
    parser.add_argument('--input_dir', type=str, help='Directory containing input NPZ files')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npz', 'visualization', 'both'],
                       help='Output format: npz, visualization, or both')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs (comma-separated, empty for CPU)')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for data loading')
    
    # Visualization options
    parser.add_argument('--vis_slices', type=int, default=5, help='Number of slices to visualize')
    parser.add_argument('--vis_channels', type=str, default='0,1,2,3', help='Channels to visualize (comma-separated)')
    
    return parser.parse_args()

def create_inference_model(opt):
    """Create and load the trained model"""
    # Import our multi-channel networks
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel
        print("✅ Successfully imported 4-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import 4-channel networks: {e}")
        raise
    
    # Parse GPU IDs
    if opt.gpu_ids:
        opt.gpu_ids = [int(id) for id in opt.gpu_ids.split(',') if id.strip()]
    else:
        opt.gpu_ids = []
    
    # Create generator with EXACT training configuration
    netG = define_G_4channel(
        input_nc=4,  # Fixed to match training
        output_nc=4,  # Fixed to match training
        ngf=64,  # From training output
        which_model_netG='unet_4channel_128',  # From training output
        norm='instance',  # IMPORTANT: training used instance norm, not batch
        use_dropout=True,  # Training had no_dropout=False, so dropout was enabled
        gpu_ids=opt.gpu_ids
    )
    
    # Load trained weights
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_path = os.path.join(checkpoint_dir, f'netG_{opt.which_epoch}.pth')
    
    if not os.path.exists(model_path):
        # Try alternative naming conventions
        possible_paths = [
            os.path.join(checkpoint_dir, f'{opt.which_epoch}_net_G.pth'),
            os.path.join(checkpoint_dir, 'netG_latest.pth'),
            os.path.join(checkpoint_dir, 'latest_net_G.pth')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            available_files = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
            raise FileNotFoundError(f"Could not find model weights. Available files: {available_files}")
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load the state dict
    if opt.gpu_ids:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    # Load with strict=False as backup
    try:
        netG.load_state_dict(state_dict, strict=True)
        print("✅ Loaded model with strict=True")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed, trying with strict=False: {e}")
        netG.load_state_dict(state_dict, strict=False)
        print("✅ Loaded model with strict=False")
    
    netG.eval()
    
    print(f"🏗️ Model loaded successfully:")
    print(f"   Input channels: 4")
    print(f"   Output channels: 4") 
    print(f"   Architecture: unet_4channel_128")
    print(f"   Normalization: instance")
    print(f"   Dropout: enabled")
    print(f"   Device: {'GPU ' + str(opt.gpu_ids) if opt.gpu_ids else 'CPU'}")
    
    return netG

def load_latent_data(file_path):
    """Load latent data from NPZ file"""
    try:
        data = np.load(file_path)
        
        # Try different possible keys for latent data
        possible_keys = ['latent_mr', 'data', 'latent', 'A', 'input']
        latent_data = None
        
        for key in possible_keys:
            if key in data:
                latent_data = data[key]
                print(f"   Found latent data with key: '{key}'")
                break
        
        if latent_data is None:
            # Show available keys
            available_keys = list(data.keys())
            raise KeyError(f"Could not find latent data. Available keys: {available_keys}")
        
        # Ensure correct shape [C, D, H, W]
        if latent_data.ndim == 3:
            # Add channel dimension if missing
            latent_data = latent_data[np.newaxis, ...]  # [1, D, H, W]
            print(f"   Added channel dimension: {latent_data.shape}")
        
        if latent_data.shape[0] != 4:
            print(f"   Warning: Expected 4 channels, got {latent_data.shape[0]}")
        
        print(f"   Loaded shape: {latent_data.shape}")
        print(f"   Data range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
        
        # Get metadata if available
        metadata = {}
        for key in data.keys():
            if key != 'latent_mr' and key != 'data' and key != 'latent':
                try:
                    metadata[key] = data[key]
                except:
                    pass
        
        return latent_data, metadata
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        raise

def process_latent(netG, latent_data, opt):
    """Process latent data through the model"""
    # Convert to torch tensor
    if isinstance(latent_data, np.ndarray):
        latent_tensor = torch.from_numpy(latent_data).float()
    else:
        latent_tensor = latent_data.float()
    
    # Add batch dimension if needed
    if latent_tensor.ndim == 4:  # [C, D, H, W]
        latent_tensor = latent_tensor.unsqueeze(0)  # [1, C, D, H, W]
    
    # Move to GPU if available
    if opt.gpu_ids:
        latent_tensor = latent_tensor.cuda(opt.gpu_ids[0])
    
    # Run inference
    with torch.no_grad():
        translated_latent = netG(latent_tensor)
    
    # Move back to CPU and convert to numpy
    translated_latent = translated_latent.cpu().numpy()
    
    # Remove batch dimension
    if translated_latent.shape[0] == 1:
        translated_latent = translated_latent[0]  # [C, D, H, W]
    
    return translated_latent

def save_results(original_latent, translated_latent, output_path, metadata, opt):
    """Save inference results"""
    base_path = os.path.splitext(output_path)[0]
    
    if opt.save_format in ['npz', 'both']:
        # Save NPZ with comprehensive data
        npz_path = f"{base_path}_translated.npz"
        
        save_data = {
            'original_latent': original_latent,
            'translated_latent': translated_latent,
            'input_shape': np.array(original_latent.shape),
            'output_shape': np.array(translated_latent.shape),
            'model_name': opt.name,
            'model_epoch': opt.which_epoch
        }
        
        # Add original metadata
        save_data.update(metadata)
        
        np.savez_compressed(npz_path, **save_data)
        print(f"💾 Saved NPZ: {npz_path}")
    
    if opt.save_format in ['visualization', 'both']:
        # Create visualization
        vis_path = f"{base_path}_visualization.png"
        create_visualization(original_latent, translated_latent, vis_path, opt)

def create_visualization(original_latent, translated_latent, output_path, opt):
    """Create visualization comparing original and translated latent representations"""
    # Parse channels to visualize
    vis_channels = [int(c.strip()) for c in opt.vis_channels.split(',')]
    vis_channels = [c for c in vis_channels if c < original_latent.shape[0]]
    
    if not vis_channels:
        print(f"⚠️ Warning: No valid channels to visualize")
        return
    
    # Calculate number of slices to show
    depth = original_latent.shape[1]  # D dimension
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, opt.vis_slices, dtype=int)
    
    # Create figure
    n_channels = len(vis_channels)
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(2 * n_channels, n_slices, figsize=(3 * n_slices, 3 * 2 * n_channels))
    
    if n_channels == 1 and n_slices == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_channels == 1:
        axes = axes.reshape(2, n_slices)
    elif n_slices == 1:
        axes = axes.reshape(2 * n_channels, 1)
    
    for ch_idx, channel in enumerate(vis_channels):
        for slice_idx, slice_pos in enumerate(slice_indices):
            # Original latent
            orig_slice = original_latent[channel, slice_pos, :, :]
            axes[ch_idx * 2, slice_idx].imshow(orig_slice, cmap='gray')
            axes[ch_idx * 2, slice_idx].set_title(f'Original Ch{channel} S{slice_pos}')
            axes[ch_idx * 2, slice_idx].axis('off')
            
            # Translated latent
            trans_slice = translated_latent[channel, slice_pos, :, :]
            axes[ch_idx * 2 + 1, slice_idx].imshow(trans_slice, cmap='gray')
            axes[ch_idx * 2 + 1, slice_idx].set_title(f'Translated Ch{channel} S{slice_pos}')
            axes[ch_idx * 2 + 1, slice_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization: {output_path}")

def find_input_files(opt):
    """Find input files to process"""
    input_files = []
    
    if opt.input_file:
        # Single file
        if os.path.exists(opt.input_file):
            input_files.append(opt.input_file)
        else:
            raise FileNotFoundError(f"Input file not found: {opt.input_file}")
    
    elif opt.input_dir:
        # Directory of files
        if not os.path.exists(opt.input_dir):
            raise FileNotFoundError(f"Input directory not found: {opt.input_dir}")
        
        # Look for NPZ files
        patterns = ['*.npz', '**/*.npz']
        for pattern in patterns:
            files = glob.glob(os.path.join(opt.input_dir, pattern), recursive=True)
            input_files.extend(files)
        
        if not input_files:
            raise FileNotFoundError(f"No NPZ files found in: {opt.input_dir}")
    
    else:
        raise ValueError("Must specify either --input_file or --input_dir")
    
    # Remove duplicates and sort
    input_files = sorted(list(set(input_files)))
    
    print(f"📁 Found {len(input_files)} input files:")
    for file in input_files[:5]:  # Show first 5
        print(f"   {file}")
    if len(input_files) > 5:
        print(f"   ... and {len(input_files) - 5} more")
    
    return input_files

def main():
    # Parse arguments
    opt = parse_inference_args()
    
    print(f"🚀 4-Channel Latent-to-Latent Inference")
    print(f"   Model: {opt.name}")
    print(f"   Epoch: {opt.which_epoch}")
    print(f"   Channels: {opt.input_nc} → {opt.output_nc}")
    print(f"   Output format: {opt.save_format}")
    
    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)
    print(f"📁 Output directory: {opt.output_dir}")
    
    # Find input files
    input_files = find_input_files(opt)
    
    # Create model
    print("\n🏗️ Loading model...")
    netG = create_inference_model(opt)
    
    # Process files
    print(f"\n🔄 Processing {len(input_files)} files...")
    start_time = time.time()
    
    for i, input_file in enumerate(tqdm(input_files, desc="Processing")):
        try:
            # Load input data
            print(f"\n📂 Processing {i+1}/{len(input_files)}: {os.path.basename(input_file)}")
            original_latent, metadata = load_latent_data(input_file)
            
            # Run inference
            translated_latent = process_latent(netG, original_latent, opt)
            
            # Prepare output path
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            output_path = os.path.join(opt.output_dir, f"{input_basename}_result")
            
            # Save results
            save_results(original_latent, translated_latent, output_path, metadata, opt)
            
            # Print stats
            print(f"   Input shape: {original_latent.shape}")
            print(f"   Output shape: {translated_latent.shape}")
            print(f"   Input range: [{original_latent.min():.3f}, {original_latent.max():.3f}]")
            print(f"   Output range: [{translated_latent.min():.3f}, {translated_latent.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Inference completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved in: {opt.output_dir}")
    print(f"🔍 Processed {len(input_files)} files")
    
    if opt.save_format in ['npz', 'both']:
        print(f"\n📦 NPZ files contain:")
        print(f"   - original_latent: Input 4-channel latent [4, D, H, W]")
        print(f"   - translated_latent: Output 4-channel latent [4, D, H, W]")
        print(f"   - model metadata and original file metadata")
    
    if opt.save_format in ['visualization', 'both']:
        print(f"\n📊 Visualization files show:")
        print(f"   - Side-by-side comparison of original vs translated latent")
        print(f"   - Multiple slices across {opt.vis_slices} depth positions")
        print(f"   - Channels: {opt.vis_channels}")

if __name__ == '__main__':
    main()
