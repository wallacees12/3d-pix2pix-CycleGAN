#!/usr/bin/env python3
"""
4-Channel Test Script for Latent-to-Latent Translation
Tests trained latent-to-latent models on test data
"""

import os
import torch
import numpy as np
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from tqdm import tqdm
import time

def create_latent_to_latent_test_model(opt):
    """Create latent-to-latent test model"""
    # Import our multi-channel networks
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel
        print("✅ Successfully imported multi-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import 4-channel networks: {e}")
        raise
    
    # Convert architecture names for 4-channel compatibility
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
            # Keep the original if we don't know how to convert it
            print(f"⚠️ Unknown architecture: {arch_name}, using as-is")
        
        print(f"🔄 Converted architecture: {opt.which_model_netG} → {arch_name}")
    
    # Create latent-to-latent generator
    netG = define_G_4channel(
        input_nc=opt.input_nc,     # Input latent channels (e.g., 4)
        output_nc=opt.output_nc,   # Output latent channels (e.g., 4)
        ngf=opt.ngf,
        which_model_netG="unet_4channel_128",
        norm=opt.norm,
        use_dropout=not opt.no_dropout,
        gpu_ids=opt.gpu_ids
    )
    
    # Load trained weights
    model_path = os.path.join(opt.checkpoints_dir, opt.name, f'netG_{opt.which_epoch}.pth')
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.which_epoch}_net_G.pth')
    
    if os.path.exists(model_path):
        print(f"📂 Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        netG.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Set to evaluation mode
    netG.eval()
    
    print(f"🏗️ Created latent-to-latent test model:")
    print(f"   Input channels: {opt.input_nc}")
    print(f"   Output channels: {opt.output_nc}")
    print(f"   Architecture: {arch_name}")
    
    return netG

def test_latent_to_latent(opt):
    """Run latent-to-latent inference"""
    
    # Create model
    print("🏗️ Creating latent-to-latent model...")
    netG = create_latent_to_latent_test_model(opt)
    
    # Create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'📊 Test dataset size: {dataset_size}')
    
    # Create output directory
    results_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}_npz')
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}")
    
    # Testing loop
    print(f"🚀 Starting latent-to-latent inference...")
    print(f"   Model: {opt.name}")
    print(f"   Epoch: {opt.which_epoch}")
    print(f"   Input channels: {opt.input_nc}")
    print(f"   Output channels: {opt.output_nc}")
    
    total_time = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Processing")):
            start_time = time.time()
            
            # Get input data
            real_A = data['A']  # Input latent (e.g., MR latent)
            A_paths = data['A_paths']
            
            # Move to GPU if available
            if opt.gpu_ids:
                real_A = real_A.cuda(opt.gpu_ids[0])
            
            # Generate latent output
            fake_B = netG(real_A)
            
            # Move back to CPU for saving
            fake_B = fake_B.cpu().numpy()
            real_A = real_A.cpu().numpy()
            
            # Save results for each sample in batch
            for j in range(real_A.shape[0]):
                # Extract filename without extension
                input_path = A_paths[j]
                filename = os.path.basename(input_path)
                
                # Create output filename
                if filename.endswith('_latent_mr.npz'):
                    # Replace _latent_mr.npz with _latent_translated.npz
                    output_filename = filename.replace('_latent_mr.npz', '_latent_translated.npz')
                else:
                    # Generic case
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}_translated.npz"
                
                output_path = os.path.join(results_dir, output_filename)
                
                # Save translated latent
                np.savez_compressed(
                    output_path,
                    translated_latent=fake_B[j],  # Shape: [output_nc, D, H, W]
                    original_latent=real_A[j],    # Shape: [input_nc, D, H, W] 
                    input_path=input_path
                )
                
                # Print progress
                if i % 10 == 0:
                    print(f"   Processed {output_filename}")
            
            batch_time = time.time() - start_time
            total_time += batch_time
    
    print(f"✅ Latent-to-latent inference completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved in: {results_dir}")
    print(f"🔍 Generated {dataset_size * opt.batchSize} translated latent representations")

def main():
    # Parse test options
    opt = TestOptions().parse()
    
    # Set defaults for latent-to-latent testing
    if not hasattr(opt, 'input_nc'):
        opt.input_nc = 4
    if not hasattr(opt, 'output_nc'):
        opt.output_nc = 4  # For latent-to-latent
    if not hasattr(opt, 'dataset_mode'):
        opt.dataset_mode = 'four_channel'
    
    # Determine the translation type
    if opt.input_nc == 4 and opt.output_nc == 4:
        print("🔄 Mode: Latent-to-Latent Translation")
    elif opt.input_nc == 4 and opt.output_nc == 1:
        print("🧠 Mode: Latent MR → Synthetic CT")
    else:
        print(f"🔧 Mode: {opt.input_nc}-channel → {opt.output_nc}-channel Translation")
    
    # Run inference
    test_latent_to_latent(opt)

if __name__ == '__main__':
    main()