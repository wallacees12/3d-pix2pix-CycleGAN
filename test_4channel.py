#!/usr/bin/env python3
"""
4-Channel 3D Pix2Pix Testing Script
Tests a trained 4-channel model and saves results in NPZ format

This script is specifically designed to work with:
- networks_3d_4channel.py (4→1 channel architectures)
- 4-channel MR latent datasets (created by 4Channel_data.ipynb)
- Models trained with train_4channel.py
"""

import time
import os
import torch
import numpy as np
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from tqdm import tqdm

def create_4channel_test_model(opt):
    """Create 4-channel model for testing"""
    # Import our 4-channel networks
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel
        print("✅ Successfully imported 4-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import 4-channel networks: {e}")
        print("   Make sure networks_3d_4channel.py is in the models/ directory")
        raise
    
    # Create 4-channel generator (4 input channels → 1 output channel)
    netG = define_G_4channel(
        input_nc=4,                           # 4-channel MR latent input
        output_nc=1,                          # Single-channel synthetic CT output
        ngf=opt.ngf,                         # Number of generator filters
        which_model_netG=opt.which_model_netG, # Architecture type
        norm=opt.norm,                       # Normalization type
        use_dropout=not opt.no_dropout,      # Dropout usage
        gpu_ids=opt.gpu_ids                  # GPU devices
    )
    
    # Load trained weights
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_path = os.path.join(checkpoint_dir, f'netG_{opt.which_epoch}.pth')
    
    if not os.path.exists(model_path):
        # Try alternative naming conventions
        alternative_paths = [
            os.path.join(checkpoint_dir, f'{opt.which_epoch}_net_G.pth'),
            os.path.join(checkpoint_dir, f'netG_epoch_{opt.which_epoch}.pth'),
            os.path.join(checkpoint_dir, 'netG_latest.pth')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Model not found. Tried:\n" + 
                                  f"  - {model_path}\n" + 
                                  "\n".join(f"  - {p}" for p in alternative_paths))
    
    print(f"📦 Loading model from: {model_path}")
    netG.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    netG.eval()
    
    # Create model wrapper for testing
    class TestModel4Channel:
        def __init__(self, netG):
            self.netG = netG
            self.gpu_ids = opt.gpu_ids
        
        def set_input(self, input):
            """Set input data for the model"""
            if 'A' in input:
                self.real_A = input['A']  # 4-channel MR latent
                if 'B' in input:
                    self.real_B = input['B']  # 1-channel CT (if available)
                else:
                    self.real_B = None
            elif 'data' in input:
                # Custom 4-channel dataset format
                self.real_A = input['data']
                self.real_B = None
            
            # Move to GPU if available
            if self.gpu_ids:
                self.real_A = self.real_A.cuda(self.gpu_ids[0])
                if self.real_B is not None:
                    self.real_B = self.real_B.cuda(self.gpu_ids[0])
        
        def test(self):
            """Run inference"""
            with torch.no_grad():
                self.fake_B = self.netG(self.real_A)
        
        def get_current_visuals(self):
            """Get current visual results"""
            visuals = {}
            
            # Convert to numpy and move to CPU
            if hasattr(self, 'real_A'):
                real_A_np = self.real_A.cpu().numpy()
                # Take first channel for visualization
                visuals['real_A'] = real_A_np[0, 0:1, :, :, :]
            
            if hasattr(self, 'fake_B'):
                visuals['fake_B'] = self.fake_B.cpu().numpy()[0, :, :, :, :]
            
            if hasattr(self, 'real_B') and self.real_B is not None:
                visuals['real_B'] = self.real_B.cpu().numpy()[0, :, :, :, :]
            
            return visuals
        
        def get_image_paths(self):
            """Get image paths"""
            return ['test_sample']
    
    return TestModel4Channel(netG)

def denormalize_ct(ct_data, method='tanh'):
    """
    Convert normalized CT data back to Hounsfield Units
    
    Args:
        ct_data: Normalized CT data (typically [-1, 1] range)
        method: Normalization method used during training
    
    Returns:
        CT data in Hounsfield Units
    """
    if method == 'tanh':
        # Convert from [-1, 1] to Hounsfield Units
        # Assuming original range was approximately [-1000, 3000] HU
        ct_hu = (ct_data + 1.0) * 2000.0 - 1000.0
    else:
        # Default: assume already in HU or similar range
        ct_hu = ct_data
    
    return ct_hu

def main():
    # Parse test options
    opt = TestOptions().parse()
    
    # Override some options for 4-channel testing
    opt.input_nc = 4   # 4-channel input
    opt.output_nc = 1  # 1-channel output
    opt.dataset_mode = 'four_channel'  # Use our 4-channel dataset loader
    opt.serial_batches = True  # No shuffling for testing
    opt.no_flip = True  # No data augmentation
    
    # Set default 4-channel architecture if not specified
    if not hasattr(opt, 'which_model_netG') or opt.which_model_netG in ['unet_128', 'unet_256']:
        opt.which_model_netG = 'unet_4channel_128'  # Use our 4-channel UNet
        print(f"🔧 Using 4-channel architecture: {opt.which_model_netG}")
    
    print(f"🔧 4-Channel Testing Configuration:")
    print(f"   📊 Input channels: {opt.input_nc}")
    print(f"   📊 Output channels: {opt.output_nc}")
    print(f"   🏗️ Generator: {opt.which_model_netG}")
    print(f"   📁 Dataset mode: {opt.dataset_mode}")
    print(f"   🎮 GPU(s): {opt.gpu_ids}")
    print(f"   💾 Model: {opt.name}")
    print(f"   📅 Epoch: {opt.which_epoch}")
    
    # Create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'📊 Dataset size: {dataset_size}')
    
    # Create model
    print("🏗️ Creating 4-channel test model...")
    model = create_4channel_test_model(opt)
    
    # Create results directory for NPZ files only
    npz_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}_npz')
    os.makedirs(npz_dir, exist_ok=True)
    
    print(f"💾 NPZ results will be saved to: {npz_dir}")
    
    # Testing loop
    print(f"🚀 Starting 4-channel inference...")
    
    for i, data in enumerate(tqdm(dataset, desc="Processing samples")):
        if i >= opt.how_many:
            break
        
        # Set input and run inference
        model.set_input(data)
        model.test()
        
        # Get results
        visuals = model.get_current_visuals()
        
        # Save NPZ results with proper denormalization
        sample_name = f"sample_{i:04d}"
        npz_path = os.path.join(npz_dir, f"{sample_name}.npz")
        
        # Prepare data for saving
        save_data = {}
        
        if 'real_A' in visuals:
            save_data['real_A'] = visuals['real_A']
        
        if 'fake_B' in visuals:
            # Denormalize synthetic CT to Hounsfield Units
            fake_B_hu = denormalize_ct(visuals['fake_B'], method='tanh')
            save_data['fake_B'] = fake_B_hu
        
        if 'real_B' in visuals:
            # Denormalize ground truth CT to Hounsfield Units
            real_B_hu = denormalize_ct(visuals['real_B'], method='tanh')
            save_data['real_B'] = real_B_hu
        
        # Save NPZ file
        np.savez_compressed(npz_path, **save_data)
        
        # Print progress
        if i % 10 == 0 or i == 0:
            print(f"📊 Processed {i+1}/{min(dataset_size, opt.how_many)} samples")
            if 'fake_B' in save_data:
                fake_B_stats = save_data['fake_B']
                print(f"   Synthetic CT range: [{fake_B_stats.min():.1f}, {fake_B_stats.max():.1f}] HU")
                print(f"   Saved: {npz_path}")
    
    # Summary
    print(f"\n✅ 4-Channel testing completed!")
    print(f"📊 Processed {min(i+1, opt.how_many)} samples")
    print(f"💾 NPZ files saved to: {npz_dir}")
    print(f"📈 Each NPZ file contains:")
    print(f"   - real_A: Input 4-channel MR latent (first channel only for viz)")
    print(f"   - fake_B: Generated synthetic CT in Hounsfield Units")
    if 'real_B' in save_data:
        print(f"   - real_B: Ground truth CT in Hounsfield Units")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Use Visualize_NPZ_Results.ipynb to inspect results")
    print(f"   2. Use Upscaling_Pipeline.ipynb for final upscaling")
    print(f"   3. Files are ready for the organized pipeline structure")

if __name__ == '__main__':
    main()
