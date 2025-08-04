#!/usr/bin/env python3
"""
Upscale Synthetic CT from NPZ files
Upscales low-resolution synthetic CT to original MR dimensions using stored metadata
Supports multiple interpolation methods and output formats
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
import time

def load_npz_data(npz_path):
    """
    Load NPZ file and extract all necessary data
    
    Args:
        npz_path: Path to NPZ file containing synthetic CT and metadata
    
    Returns:
        dict with keys: fake_B, target_shape, spacing, origin, sample_name, etc.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract required data
    result = {
        'fake_B': data['fake_B'],  # Synthetic CT [D_low, H_low, W_low]
        'target_shape': data['target_shape'],  # Target dimensions [D_high, H_high, W_high]
        'sample_name': str(data['sample_name']) if 'sample_name' in data else 'unknown',
        'latent_shape': data.get('latent_shape', data['fake_B'].shape),
    }
    
    # Optional metadata
    if 'spacing' in data:
        result['spacing'] = tuple(data['spacing'])
    else:
        result['spacing'] = (1.0, 1.0, 1.0)  # Default 1mm spacing
    
    if 'origin' in data:
        result['origin'] = tuple(data['origin'])
    else:
        result['origin'] = (0.0, 0.0, 0.0)  # Default origin
    
    if 'original_mr' in data:
        result['original_mr'] = data['original_mr']
    
    # Calculate scale factors
    target_shape = result['target_shape']
    latent_shape = result['latent_shape']
    
    scale_factors = [target_shape[i] / latent_shape[i] for i in range(3)]
    result['scale_factors'] = scale_factors
    
    print(f"📂 Loaded NPZ: {os.path.basename(npz_path)}")
    print(f"   Sample: {result['sample_name']}")
    print(f"   Latent shape: {latent_shape}")
    print(f"   Target shape: {target_shape}")
    print(f"   Scale factors (D,H,W): {scale_factors}")
    print(f"   Spacing: {result['spacing']}")
    print(f"   Origin: {result['origin']}")
    
    return result

def upscale_trilinear_torch(volume, target_shape, device='cpu'):
    """
    Upscale volume using PyTorch trilinear interpolation
    
    Args:
        volume: Input volume [D, H, W]
        target_shape: Target dimensions [D_target, H_target, W_target]
        device: 'cpu' or 'cuda'
    
    Returns:
        Upscaled volume as numpy array
    """
    # Convert to torch tensor and add batch and channel dimensions
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # Move to device
    volume_tensor = volume_tensor.to(device)
    
    # Perform trilinear interpolation
    upscaled_tensor = F.interpolate(
        volume_tensor,
        size=target_shape,
        mode='trilinear',
        align_corners=False
    )
    
    # Move back to CPU and convert to numpy
    upscaled_volume = upscaled_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    return upscaled_volume

def upscale_scipy_zoom(volume, scale_factors, order=1):
    """
    Upscale volume using scipy zoom (spline interpolation)
    
    Args:
        volume: Input volume [D, H, W]
        scale_factors: Scale factors for each dimension [scale_d, scale_h, scale_w]
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Upscaled volume as numpy array
    """
    upscaled_volume = ndimage.zoom(volume, scale_factors, order=order)
    return upscaled_volume

def upscale_sitk_resample(volume, target_shape, spacing, origin, interpolation='linear'):
    """
    Upscale volume using SimpleITK resampling with proper medical image handling
    
    Args:
        volume: Input volume [D, H, W]
        target_shape: Target dimensions [D_target, H_target, W_target]
        spacing: Voxel spacing (z, y, x)
        origin: Image origin (z, y, x)
        interpolation: 'linear', 'nearest', 'bspline', or 'gaussian'
    
    Returns:
        Upscaled volume as numpy array
    """
    # Convert numpy array to SimpleITK image
    image = sitk.GetImageFromArray(volume)
    
    # Calculate input spacing (assuming the latent image has isotropic spacing derived from scale factors)
    input_shape = volume.shape
    # Calculate spacing for the latent image based on target spacing and scale factors
    scale_factors = [target_shape[i] / input_shape[i] for i in range(3)]
    input_spacing = [spacing[i] * scale_factors[i] for i in range(3)]
    
    # Set input image properties
    # Convert from (z, y, x) to (x, y, z) for SimpleITK
    image.SetSpacing([input_spacing[2], input_spacing[1], input_spacing[0]])
    image.SetOrigin([origin[2], origin[1], origin[0]])
    
    # Set up the resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing[2], spacing[1], spacing[0]])  # Convert to (x, y, z)
    resampler.SetSize([target_shape[2], target_shape[1], target_shape[0]])  # Convert to (x, y, z)
    resampler.SetOutputOrigin([origin[2], origin[1], origin[0]])  # Convert to (x, y, z)
    resampler.SetOutputDirection(image.GetDirection())
    
    # Set interpolation method
    if interpolation.lower() == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation.lower() == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation.lower() == 'bspline':
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation.lower() == 'gaussian':
        resampler.SetInterpolator(sitk.sitkGaussian)
    else:
        print(f"⚠️ Unknown interpolation method '{interpolation}', using linear")
        resampler.SetInterpolator(sitk.sitkLinear)
    
    # Perform resampling
    upscaled_image = resampler.Execute(image)
    
    # Convert back to numpy array
    upscaled_volume = sitk.GetArrayFromImage(upscaled_image)
    
    return upscaled_volume

def save_upscaled_result(upscaled_ct, sample_name, spacing, origin, output_dir, 
                        save_format='mha', include_metadata=True):
    """
    Save upscaled synthetic CT
    
    Args:
        upscaled_ct: Upscaled CT volume [D, H, W]
        sample_name: Patient identifier
        spacing: Voxel spacing (z, y, x)
        origin: Image origin (z, y, x)
        output_dir: Output directory
        save_format: 'mha', 'nii', or 'npz'
        include_metadata: Whether to include spacing/origin in medical formats
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_format.lower() == 'npz':
        # Save as NPZ for further processing
        output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.npz")
        np.savez_compressed(output_file,
                           upscaled_ct=upscaled_ct,
                           spacing=np.array(spacing),
                           origin=np.array(origin),
                           sample_name=sample_name,
                           shape=np.array(upscaled_ct.shape))
        
    elif save_format.lower() in ['mha', 'nii', 'nii.gz']:
        # Save as medical image format
        if save_format.lower() == 'mha':
            output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.mha")
        elif save_format.lower() == 'nii':
            output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.nii")
        else:  # nii.gz
            output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.nii.gz")
        
        # Convert to SimpleITK image
        image = sitk.GetImageFromArray(upscaled_ct)
        
        if include_metadata:
            # Set spacing and origin (convert from D,H,W to x,y,z)
            image.SetSpacing([spacing[2], spacing[1], spacing[0]])
            image.SetOrigin([origin[2], origin[1], origin[0]])
        
        # Write the image
        sitk.WriteImage(image, output_file)
    
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    
    print(f"💾 Saved upscaled CT: {output_file}")
    print(f"   Shape: {upscaled_ct.shape}")
    print(f"   HU range: [{upscaled_ct.min():.1f}, {upscaled_ct.max():.1f}]")
    
    return output_file

def upscale_npz_file(npz_path, output_dir, method='trilinear', device='cpu', 
                    save_format='mha', interpolation_order=1):
    """
    Process a single NPZ file
    
    Args:
        npz_path: Path to input NPZ file
        output_dir: Output directory
        method: Upscaling method ('trilinear', 'scipy', 'sitk')
        device: 'cpu' or 'cuda' for torch methods
        save_format: Output format ('mha', 'nii', 'npz')
        interpolation_order: Order for scipy interpolation
    
    Returns:
        Path to output file
    """
    # Load NPZ data
    data = load_npz_data(npz_path)
    
    synthetic_ct = data['fake_B']
    target_shape = data['target_shape']
    scale_factors = data['scale_factors']
    spacing = data['spacing']
    origin = data['origin']
    sample_name = data['sample_name']
    
    print(f"\n🔧 Upscaling using {method} method...")
    start_time = time.time()
    
    # Perform upscaling based on method
    if method.lower() == 'trilinear':
        upscaled_ct = upscale_trilinear_torch(synthetic_ct, target_shape, device)
        
    elif method.lower() == 'scipy':
        upscaled_ct = upscale_scipy_zoom(synthetic_ct, scale_factors, order=interpolation_order)
        
    elif method.lower() == 'sitk':
        upscaled_ct = upscale_sitk_resample(synthetic_ct, target_shape, spacing, origin, 
                                          interpolation='linear')
    else:
        raise ValueError(f"Unknown upscaling method: {method}")
    
    upscale_time = time.time() - start_time
    print(f"⏱️ Upscaling completed in {upscale_time:.2f} seconds")
    
    # Verify target shape
    if tuple(upscaled_ct.shape) != tuple(target_shape):
        print(f"⚠️ Warning: Output shape {upscaled_ct.shape} doesn't match target {target_shape}")
    
    # Save result
    output_file = save_upscaled_result(upscaled_ct, sample_name, spacing, origin, 
                                     output_dir, save_format)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Upscale synthetic CT from NPZ files')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input NPZ file or directory containing NPZ files')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for upscaled results')
    parser.add_argument('--method', '-m', type=str, default='trilinear',
                       choices=['trilinear', 'scipy', 'sitk'],
                       help='Upscaling method (default: trilinear)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for torch operations (default: cpu)')
    parser.add_argument('--format', '-f', type=str, default='mha',
                       choices=['mha', 'nii', 'nii.gz', 'npz'],
                       help='Output format (default: mha)')
    parser.add_argument('--order', type=int, default=1,
                       choices=[0, 1, 2, 3],
                       help='Interpolation order for scipy method (0=nearest, 1=linear, 3=cubic)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all NPZ files in input directory')
    
    args = parser.parse_args()
    
    print("🚀 Synthetic CT Upscaling Tool")
    print(f"   Method: {args.method}")
    print(f"   Device: {args.device}")
    print(f"   Output format: {args.format}")
    print(f"   Interpolation order: {args.order} (scipy only)")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Determine input files
    if os.path.isfile(args.input):
        if not args.input.endswith('.npz'):
            raise ValueError("Input file must be an NPZ file")
        npz_files = [args.input]
        
    elif os.path.isdir(args.input):
        if not args.batch:
            raise ValueError("Use --batch flag to process directory")
        npz_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                    if f.endswith('.npz') and 'synthetic_ct' in f]
        if not npz_files:
            raise ValueError(f"No synthetic CT NPZ files found in {args.input}")
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    print(f"📁 Found {len(npz_files)} NPZ file(s) to process")
    
    # Process files
    total_start_time = time.time()
    
    for i, npz_file in enumerate(tqdm(npz_files, desc="Processing files")):
        print(f"\n📋 Processing {i+1}/{len(npz_files)}: {os.path.basename(npz_file)}")
        
        try:
            output_file = upscale_npz_file(
                npz_path=npz_file,
                output_dir=args.output,
                method=args.method,
                device=args.device,
                save_format=args.format,
                interpolation_order=args.order
            )
            
        except Exception as e:
            print(f"❌ Error processing {npz_file}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\n✅ Upscaling completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved in: {args.output}")
    
    print(f"\n🔧 Usage examples:")
    print(f"   Single file: python upscale_synthetic_ct.py -i result.npz -o ./upscaled/")
    print(f"   Batch mode:  python upscale_synthetic_ct.py -i ./results/ -o ./upscaled/ --batch")
    print(f"   GPU mode:    python upscale_synthetic_ct.py -i result.npz -o ./upscaled/ --device cuda")
    print(f"   NIfTI out:   python upscale_synthetic_ct.py -i result.npz -o ./upscaled/ -f nii.gz")

if __name__ == '__main__':
    main()
