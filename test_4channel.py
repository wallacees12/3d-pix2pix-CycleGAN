#!/usr/bin/env python3
"""
Simple 4-Channel Test Script for Latent MR → Synthetic CT
Works with FourChannelTestDataset that loads both latent MR and original MR
Outputs NPZ files with synthetic CT and target dimensions for upscaling
"""

import time
import os
import torch
import numpy as np
import SimpleITK as sitk
from options.test_options import TestOptions
from tqdm import tqdm

def create_4channel_test_model(opt):
    """Create 4-channel model for testing"""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel
        print("✅ Successfully imported 4-channel networks")
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
            arch_name = 'resnet_4channel_9blocks'  # Default
        
        print(f"🔄 Converted architecture: {opt.which_model_netG} → {arch_name}")
    
    # Create 4-channel generator
    netG = define_G_4channel(
        input_nc=4,
        output_nc=1,
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
            raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir}")
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location='cpu')
    netG.load_state_dict(state_dict)
    
    # Set to evaluation mode
    netG.eval()
    
    return netG

def denormalize_ct(tensor):
    """
    Denormalize CT values from [-1, 1] back to HU range
    Assumes the model was trained with CT normalized to [-1, 1] from [-1024, 1200] HU
    """
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    
    # Convert to HU range [-1024, 1200]
    tensor = tensor * (1200 - (-1024)) + (-1024)
    
    return tensor

def save_synthetic_ct(synthetic_ct, sample_name, target_shape, real_A_sample, original_mr, 
                     result_dir, save_type='npz', spacing=None, origin=None):
    """
    Save synthetic CT in specified format
    
    Args:
        synthetic_ct: Generated CT data [D, H, W] in HU
        sample_name: Patient identifier
        target_shape: Target dimensions for upscaling
        real_A_sample: Input latent MR data [4, D, H, W]
        original_mr: Original MR data
        result_dir: Output directory
        save_type: 'npz' or 'mha'
        spacing: Tuple of (z_spacing, y_spacing, x_spacing) from original MR
        origin: Tuple of (z_origin, y_origin, x_origin) from original MR
    
    Returns:
        Path to saved file
    """
    if save_type == 'mha':
        # Save as MHA medical image format
        output_file = os.path.join(result_dir, f"{sample_name}_synthetic_ct.mha")
        
        # Convert numpy array to SimpleITK image
        synthetic_ct_image = sitk.GetImageFromArray(synthetic_ct)
        
        # Set spacing from original MR if available, otherwise use default
        if spacing is not None:
            # Convert from (z, y, x) to (x, y, z) for SimpleITK
            sitk_spacing = [spacing[2], spacing[1], spacing[0]]
            synthetic_ct_image.SetSpacing(sitk_spacing)
            print(f"   Using original MR spacing: {spacing} (D,H,W)")
        else:
            synthetic_ct_image.SetSpacing([1.0, 1.0, 1.0])  # Default 1mm spacing
            print(f"   Using default spacing: [1.0, 1.0, 1.0]")
        
        # Set origin from original MR if available, otherwise use default
        if origin is not None:
            # Convert from (z, y, x) to (x, y, z) for SimpleITK
            sitk_origin = [origin[2], origin[1], origin[0]]
            synthetic_ct_image.SetOrigin(sitk_origin)
            print(f"   Using original MR origin: {origin} (D,H,W)")
        else:
            synthetic_ct_image.SetOrigin([0.0, 0.0, 0.0])
            print(f"   Using default origin: [0.0, 0.0, 0.0]")
        
        # Write the image
        sitk.WriteImage(synthetic_ct_image, output_file)
        
        print(f"💾 Saved MHA: {output_file}")
        print(f"   Sample: {sample_name}")
        print(f"   Synthetic CT shape: {synthetic_ct.shape}")
        print(f"   HU range: [{synthetic_ct.min():.1f}, {synthetic_ct.max():.1f}]")
        
        return output_file
        
    else:  # npz format
        # Save comprehensive NPZ with all data needed for upscaling and analysis
        output_file = os.path.join(result_dir, f"{sample_name}_synthetic_ct.npz")
        
        # Prepare save data dictionary
        save_data = {
            'fake_B': synthetic_ct,                    # Generated synthetic CT [D_low, H_low, W_low] in HU
            'real_A': real_A_sample[0],               # First channel of latent MR [D_low, H_low, W_low]
            'target_shape': target_shape,             # Target dimensions [D_high, H_high, W_high] from original MR
            'original_mr_shape': np.array(original_mr.shape) if original_mr.size > 1 else target_shape,  # Original MR shape for reference
            'original_mr': original_mr,               # Full original MR data
            'sample_name': sample_name,               # Patient name
            'latent_shape': np.array(synthetic_ct.shape)  # Latent/synthetic CT shape
        }
        
        # Add spacing and origin if available
        if spacing is not None:
            save_data['spacing'] = np.array(spacing)
            print(f"   Including spacing: {spacing} (D,H,W)")
        
        if origin is not None:
            save_data['origin'] = np.array(origin)
            print(f"   Including origin: {origin} (D,H,W)")
        
        # Save the NPZ file
        np.savez_compressed(output_file, **save_data)
        
        print(f"💾 Saved NPZ: {output_file}")
        print(f"   Sample: {sample_name}")
        print(f"   Synthetic CT shape: {synthetic_ct.shape}")
        print(f"   Target shape: {target_shape}")
        print(f"   HU range: [{synthetic_ct.min():.1f}, {synthetic_ct.max():.1f}]")
        
        # Show scale factor
        if len(target_shape) == 3 and len(synthetic_ct.shape) == 3:
            scale_factors = [target_shape[i] / synthetic_ct.shape[i] for i in range(3)]
            print(f"   Scale factors: {scale_factors}")
        
        return output_file

def main():
    # Parse command line options
    opt = TestOptions().parse()
    
    # Add custom save_type argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_type', type=str, default='npz', choices=['npz', 'mha'], 
                       help='Output format: npz (comprehensive) or mha (medical image format)')
    
    # Parse only our custom argument from remaining args
    import sys
    save_args, _ = parser.parse_known_args()
    opt.save_type = save_args.save_type
    
    # Force test mode settings
    opt.phase = 'test'
    opt.serial_batches = True
    opt.no_flip = True
    opt.nThreads = 1
    
    print(f"🧪 4-Channel Testing with Original MR Dimensions:")
    print(f"   Model: {opt.name}")
    print(f"   Epoch: {opt.which_epoch}")
    print(f"   Data: {opt.dataroot}")
    print(f"   Device: {'GPU ' + str(opt.gpu_ids) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else 'CPU'}")
    print(f"   Batch size: {opt.batchSize} (will be forced to 1 for variable-sized data)")
    print(f"   Save format: {opt.save_type.upper()}")
    
    # Create test dataset using our FourChannelTestDataset
    from data.four_channel_dataset_test import FourChannelTestDataLoaderWrapper
    
    # Create dataset wrapper
    dataset_wrapper = FourChannelTestDataLoaderWrapper(opt)
    dataset = dataset_wrapper.load_data()
    
    print(f"📊 Dataset size: {len(dataset_wrapper)}")
    
    # Create model
    print("🏗️ Creating 4-channel test model...")
    netG = create_4channel_test_model(opt)
    
    # Setup result directory
    format_suffix = 'mha' if opt.save_type == 'mha' else 'npz'
    result_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}_{format_suffix}')
    os.makedirs(result_dir, exist_ok=True)
    print(f"💾 {opt.save_type.upper()} results will be saved to: {result_dir}")
    
    # Start inference
    print("🚀 Starting 4-channel inference...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Processing samples")):
            # Get input data
            real_A = data['A']  # 4-channel MR latent
            sample_names = data['sample_name']  # Patient names
            target_shapes = data['target_shape']  # Target dimensions from original MR
            spacings = data['spacing']  # Spacing from original MR
            origins = data['origin']  # Origin from original MR
            original_mrs = data['original_mr']  # Original MR data
            
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
            
            # Generate synthetic CT
            fake_B = netG(real_A)
            
            # Move back to CPU and convert to numpy
            fake_B_np = fake_B.cpu().numpy()
            real_A_np = real_A.cpu().numpy()
            
            # Process each sample in the batch
            for j in range(fake_B_np.shape[0]):
                # Get single sample data
                synthetic_ct = fake_B_np[j]  # Shape: [1, D, H, W]
                sample_name = sample_names[j] if isinstance(sample_names, (list, tuple)) else sample_names
                target_shape = target_shapes[j].numpy() if hasattr(target_shapes[j], 'numpy') else target_shapes[j]
                spacing = spacings[j] if isinstance(spacings, (list, tuple)) else spacings
                origin = origins[j] if isinstance(origins, (list, tuple)) else origins
                original_mr = original_mrs[j].numpy() if hasattr(original_mrs[j], 'numpy') else original_mrs[j]
                real_A_sample = real_A_np[j]  # Shape: [4, D, H, W]
                
                # Remove channel dimension from synthetic CT
                synthetic_ct = synthetic_ct[0]  # Shape: [D, H, W]
                
                # Denormalize to HU values
                synthetic_ct = denormalize_ct(torch.from_numpy(synthetic_ct)).numpy()
                
                # Save using specified format
                output_file = save_synthetic_ct(
                    synthetic_ct=synthetic_ct,
                    sample_name=sample_name,
                    target_shape=target_shape,
                    real_A_sample=real_A_sample,
                    original_mr=original_mr,
                    result_dir=result_dir,
                    save_type=opt.save_type,
                    spacing=spacing,
                    origin=origin
                )
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Results saved in: {result_dir}")
    print(f"🎯 Processed {len(dataset_wrapper)} samples")
    
    if opt.save_type == 'mha':
        print(f"\n📦 Each MHA file contains:")
        print(f"   - Synthetic CT in HU with original MR spacing and origin")
        print(f"   - Standard medical image format compatible with imaging software")
        print(f"\n🔧 Next steps:")
        print(f"   1. Load MHA files in medical imaging software (3D Slicer, ITK-SNAP, etc.)")
        print(f"   2. Use medical image processing tools for upscaling or analysis")
    else:
        print(f"\n📦 Each NPZ file contains:")
        print(f"   - fake_B: Synthetic CT in HU [D_low, H_low, W_low]")
        print(f"   - real_A: First channel of latent MR for reference")
        print(f"   - target_shape: Target dimensions from original MR")
        print(f"   - spacing: Voxel spacing from original MR (D,H,W)")
        print(f"   - origin: Image origin from original MR (D,H,W)")
        print(f"   - original_mr: Full original MR data")
        print(f"   - sample_name: Patient identifier")
        print(f"   - latent_shape: Shape of synthetic CT")
        print(f"\n🔧 Next steps:")
        print(f"   1. Use upscale.py to upscale synthetic CT to target dimensions")
        print(f"   2. Example: python upscale.py --file {result_dir}/{sample_name}_synthetic_ct.npz")

if __name__ == '__main__':
    main()