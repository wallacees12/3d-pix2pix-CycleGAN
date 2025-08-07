#!/usr/bin/env python3
"""
4-Channel Latent-to-Latent Inference Script
Run inference with your trained 4-channel latent-to-latent Pix2Pix model

Usage:
    python run_4channel_inference.py --name experiment_name --dataroot path/to/test/data
"""

import os
import sys
import time
import torch
import numpy as np
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from tqdm import tqdm

def create_4channel_inference_model(opt):
    """Create 4-channel model for inference"""
    # Import our multi-channel networks
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel
        print("✅ Successfully imported 4-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import 4-channel networks: {e}")
        raise
    
    # Convert architecture names for 4-channel compatibility if needed
    arch_name = opt.which_model_netG
    if not arch_name.endswith('_4channel'):
        if 'resnet_9blocks' in arch_name:
            arch_name = 'resnet_4channel_9blocks'
        elif 'resnet_6blocks' in arch_name:
            arch_name = 'resnet_4channel_6blocks'
        elif 'unet_128' in arch_name:
            arch_name = 'unet_4channel_128'
        elif 'unet_256' in arch_name:
            arch_name = 'unet_4channel_256'
        else:
            arch_name = 'resnet_4channel_9blocks'  # Default
        
        print(f"🔄 Converted architecture: {opt.which_model_netG} → {arch_name}")
    
    # Create 4-channel generator
    netG = define_G_4channel(
        input_nc=opt.input_nc,
        output_nc=opt.output_nc,
        ngf=opt.ngf,
        which_model_netG=arch_name,
        norm=opt.norm,
        use_dropout=not opt.no_dropout,
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
            raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir}. Available files: {available_files}")
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load the state dict
    if opt.gpu_ids:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    netG.load_state_dict(state_dict)
    netG.eval()
    
    print(f"🏗️ Model loaded successfully:")
    print(f"   Input channels: {opt.input_nc}")
    print(f"   Output channels: {opt.output_nc}")
    print(f"   Architecture: {arch_name}")
    
    return netG

def main():
    # Parse command line options
    opt = TestOptions().parse()
    
    # Override some defaults for 4-channel inference
    opt.input_nc = getattr(opt, 'input_nc', 4)
    opt.output_nc = getattr(opt, 'output_nc', 4)  # For latent-to-latent
    opt.dataset_mode = getattr(opt, 'dataset_mode', 'four_channel')
    opt.which_model_netG = getattr(opt, 'which_model_netG', 'unet_4channel_128')
    
    # Force test mode settings
    opt.phase = 'test'
    opt.serial_batches = True
    opt.no_flip = True
    opt.nThreads = 1
    
    print(f"🚀 4-Channel Latent-to-Latent Inference:")
    print(f"   Model: {opt.name}")
    print(f"   Epoch: {opt.which_epoch}")
    print(f"   Data: {opt.dataroot}")
    print(f"   Input channels: {opt.input_nc}")
    print(f"   Output channels: {opt.output_nc}")
    print(f"   Architecture: {opt.which_model_netG}")
    print(f"   Device: {'GPU ' + str(opt.gpu_ids) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else 'CPU'}")
    
    # Create test dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    
    print(f"📊 Dataset size: {dataset_size}")
    
    # Create model
    print("🏗️ Creating 4-channel inference model...")
    netG = create_4channel_inference_model(opt)
    
    # Setup result directory
    result_dir = os.path.join(opt.results_dir, opt.name, f'latent_to_latent_{opt.which_epoch}')
    os.makedirs(result_dir, exist_ok=True)
    print(f"💾 Results will be saved to: {result_dir}")
    
    # Start inference
    print("🔄 Starting 4-channel latent-to-latent inference...")
    
    total_time = 0
    processed_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Processing samples")):
            start_time = time.time()
            
            # Get input data
            if 'A' in data:
                real_A = data['A']  # 4-channel input latent
                A_paths = data['A_paths']
            elif 'data' in data:
                real_A = data['data']  # Alternative data format
                A_paths = data.get('paths', [f'sample_{i}'])
            else:
                print(f"⚠️ Warning: Unknown data format for batch {i}")
                continue
            
            # Move to GPU if available and requested
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                real_A = real_A.cuda()
                # Ensure model is also on GPU
                if not next(netG.parameters()).is_cuda:
                    netG = netG.cuda()
            else:
                # Ensure model is on CPU
                if next(netG.parameters()).is_cuda:
                    netG = netG.cpu()
            
            # Generate translated latent
            fake_B = netG(real_A)
            
            # Move back to CPU and convert to numpy
            fake_B_np = fake_B.cpu().numpy()
            real_A_np = real_A.cpu().numpy()
            
            # Process each sample in the batch
            for j in range(fake_B_np.shape[0]):
                # Get single sample data
                translated_latent = fake_B_np[j]  # Shape: [4, D, H, W]
                original_latent = real_A_np[j]    # Shape: [4, D, H, W]
                
                # Get sample name/path
                if isinstance(A_paths, (list, tuple)):
                    input_path = A_paths[j] if j < len(A_paths) else f'sample_{i}_{j}'
                else:
                    input_path = A_paths
                
                # Extract filename for output
                if isinstance(input_path, str):
                    filename = os.path.basename(input_path)
                    sample_name = os.path.splitext(filename)[0]
                else:
                    sample_name = f'sample_{i}_{j}'
                
                # Create output filename
                output_file = os.path.join(result_dir, f"{sample_name}_latent_translated.npz")
                
                # Save comprehensive NPZ with all relevant data
                save_data = {
                    'translated_latent': translated_latent,     # Output 4-channel latent [4, D, H, W]
                    'original_latent': original_latent,         # Input 4-channel latent [4, D, H, W]
                    'sample_name': sample_name,                 # Sample identifier
                    'input_path': str(input_path),              # Original input path
                    'model_name': opt.name,                     # Model used for inference
                    'model_epoch': opt.which_epoch,             # Epoch used
                    'input_shape': np.array(original_latent.shape),    # Input dimensions
                    'output_shape': np.array(translated_latent.shape)  # Output dimensions
                }
                
                # Save the NPZ file
                np.savez_compressed(output_file, **save_data)
                
                # Print sample info
                if processed_count < 5 or processed_count % 10 == 0:
                    print(f"   Processed: {sample_name}")
                    print(f"     Input shape: {original_latent.shape}")
                    print(f"     Output shape: {translated_latent.shape}")
                    print(f"     Input range: [{original_latent.min():.3f}, {original_latent.max():.3f}]")
                    print(f"     Output range: [{translated_latent.min():.3f}, {translated_latent.max():.3f}]")
                    print(f"     Saved: {output_file}")
                
                processed_count += 1
            
            batch_time = time.time() - start_time
            total_time += batch_time
    
    print(f"\n✅ 4-Channel latent-to-latent inference completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved in: {result_dir}")
    print(f"🔍 Processed {processed_count} samples")
    
    print(f"\n📦 Each NPZ file contains:")
    print(f"   - translated_latent: Output 4-channel latent [4, D, H, W]")
    print(f"   - original_latent: Input 4-channel latent [4, D, H, W]")
    print(f"   - sample_name: Sample identifier")
    print(f"   - input_path: Original input file path")
    print(f"   - model metadata: model_name, model_epoch")
    print(f"   - shape information: input_shape, output_shape")
    
    print(f"\n🔧 Next steps:")
    print(f"   1. Analyze translated latent representations")
    print(f"   2. Compare with original latent features")
    print(f"   3. Use translated latents for downstream tasks")
    print(f"   4. Visualize latent space transformations")

if __name__ == '__main__':
    main()
