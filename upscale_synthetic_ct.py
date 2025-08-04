#!/usr/bin/env python3
"""
Simple Upscaling Script for Synthetic CT
Takes NPZ file with synthetic CT and original MR file path
Uses SimpleITK resampling to match original MR dimensions exactly
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import time

def load_original_mr(mr_file_path):
    """
    Load original MR file and extract metadata
    
    Args:
        mr_file_path: Path to original MR file (.mha, .nii, .nii.gz)
    
    Returns:
        dict with spacing, origin, size, and direction
    """
    if not os.path.exists(mr_file_path):
        raise FileNotFoundError(f"Original MR file not found: {mr_file_path}")
    
    print(f"📂 Loading original MR: {os.path.basename(mr_file_path)}")
    
    # Load MR image
    mr_image = sitk.ReadImage(mr_file_path)
    
    # Extract metadata
    metadata = {
        'spacing': mr_image.GetSpacing(),  # (x, y, z)
        'origin': mr_image.GetOrigin(),    # (x, y, z)
        'size': mr_image.GetSize(),        # (x, y, z)
        'direction': mr_image.GetDirection()
    }
    
    # Convert size to numpy format (z, y, x) for consistency with numpy arrays
    target_shape = (metadata['size'][2], metadata['size'][1], metadata['size'][0])
    
    print(f"   Size: {metadata['size']} (x,y,z)")
    print(f"   Target shape: {target_shape} (z,y,x)")
    print(f"   Spacing: {metadata['spacing']} (x,y,z)")
    print(f"   Origin: {metadata['origin']} (x,y,z)")
    
    return metadata, target_shape

def load_synthetic_ct(npz_path):
    """
    Load synthetic CT from NPZ file
    
    Args:
        npz_path: Path to NPZ file containing synthetic CT
    
    Returns:
        tuple: (synthetic_ct_array, sample_name)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Get synthetic CT data
    if 'fake_B' in data:
        synthetic_ct = data['fake_B']
    elif 'synthetic_ct' in data:
        synthetic_ct = data['synthetic_ct']
    else:
        # Try first available array
        available_keys = [k for k in data.keys() if isinstance(data[k], np.ndarray) and data[k].ndim == 3]
        if not available_keys:
            raise ValueError(f"No 3D array found in NPZ file: {list(data.keys())}")
        synthetic_ct = data[available_keys[0]]
        print(f"⚠️ Using key '{available_keys[0]}' as synthetic CT")
    
    # Get sample name
    sample_name = str(data.get('sample_name', 'unknown'))
    
    print(f"📂 Loaded synthetic CT: {os.path.basename(npz_path)}")
    print(f"   Sample: {sample_name}")
    print(f"   Shape: {synthetic_ct.shape}")
    print(f"   HU range: [{synthetic_ct.min():.1f}, {synthetic_ct.max():.1f}]")
    
    return synthetic_ct, sample_name

def upscale_with_sitk(synthetic_ct, mr_metadata, target_shape, interpolation='linear'):
    """
    Upscale synthetic CT using SimpleITK to match original MR exactly
    
    Args:
        synthetic_ct: Input synthetic CT volume [D, H, W]
        mr_metadata: Metadata from original MR (spacing, origin, direction)
        target_shape: Target dimensions (D, H, W)
        interpolation: Interpolation method
    
    Returns:
        Upscaled synthetic CT volume
    """
    print(f"🔧 Upscaling using SimpleITK resampling...")
    print(f"   From: {synthetic_ct.shape} → To: {target_shape}")
    
    # Convert synthetic CT to SimpleITK image
    synthetic_image = sitk.GetImageFromArray(synthetic_ct)
    
    # Calculate spacing for the synthetic CT
    # Assume the synthetic CT has isotropic spacing that scales with the size difference
    scale_factors = [target_shape[i] / synthetic_ct.shape[i] for i in range(3)]
    
    # Set spacing for synthetic image (convert from z,y,x to x,y,z for SimpleITK)
    synthetic_spacing = [
        mr_metadata['spacing'][0] * scale_factors[2],  # x spacing
        mr_metadata['spacing'][1] * scale_factors[1],  # y spacing
        mr_metadata['spacing'][2] * scale_factors[0]   # z spacing
    ]
    
    synthetic_image.SetSpacing(synthetic_spacing)
    synthetic_image.SetOrigin(mr_metadata['origin'])
    synthetic_image.SetDirection(mr_metadata['direction'])
    
    print(f"   Synthetic spacing: {synthetic_spacing} (x,y,z)")
    print(f"   Target spacing: {mr_metadata['spacing']} (x,y,z)")
    print(f"   Scale factors: {[f'{f:.3f}' for f in scale_factors]} (z,y,x)")
    
    # Set up resampling to match original MR exactly
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(mr_metadata['spacing'])
    resampler.SetSize(mr_metadata['size'])  # (x, y, z)
    resampler.SetOutputOrigin(mr_metadata['origin'])
    resampler.SetOutputDirection(mr_metadata['direction'])
    
    # Set interpolation method
    if interpolation.lower() == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation.lower() == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation.lower() == 'bspline':
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        print(f"⚠️ Unknown interpolation '{interpolation}', using linear")
        resampler.SetInterpolator(sitk.sitkLinear)
    
    # Perform resampling
    start_time = time.time()
    upscaled_image = resampler.Execute(synthetic_image)
    upscale_time = time.time() - start_time
    
    print(f"⏱️ Resampling completed in {upscale_time:.2f} seconds")
    
    # Convert back to numpy array
    upscaled_volume = sitk.GetArrayFromImage(upscaled_image)
    
    return upscaled_volume

def save_upscaled_ct(upscaled_ct, sample_name, mr_metadata, output_dir, save_format='mha'):
    """
    Save upscaled synthetic CT with original MR metadata
    
    Args:
        upscaled_ct: Upscaled CT volume
        sample_name: Patient identifier
        mr_metadata: Original MR metadata
        output_dir: Output directory
        save_format: Output format ('mha', 'nii', 'nii.gz')
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename
    if save_format.lower() == 'mha':
        output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.mha")
    elif save_format.lower() == 'nii':
        output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.nii")
    elif save_format.lower() == 'nii.gz':
        output_file = os.path.join(output_dir, f"{sample_name}_upscaled_ct.nii.gz")
    else:
        raise ValueError(f"Unsupported format: {save_format}")
    
    # Convert to SimpleITK image with original MR metadata
    upscaled_image = sitk.GetImageFromArray(upscaled_ct)
    upscaled_image.SetSpacing(mr_metadata['spacing'])
    upscaled_image.SetOrigin(mr_metadata['origin'])
    upscaled_image.SetDirection(mr_metadata['direction'])
    
    # Save the image
    sitk.WriteImage(upscaled_image, output_file)
    
    print(f"💾 Saved upscaled CT: {output_file}")
    print(f"   Shape: {upscaled_ct.shape}")
    print(f"   HU range: [{upscaled_ct.min():.1f}, {upscaled_ct.max():.1f}]")
    print(f"   Spacing: {mr_metadata['spacing']} (x,y,z)")
    print(f"   Origin: {mr_metadata['origin']} (x,y,z)")
    
    return output_file

def process_file(npz_path, mr_path, output_dir, save_format='mha', interpolation='linear'):
    """
    Process a single NPZ file with its corresponding MR file
    
    Args:
        npz_path: Path to NPZ file with synthetic CT
        mr_path: Path to original MR file
        output_dir: Output directory
        save_format: Output format
        interpolation: Interpolation method
    
    Returns:
        Path to output file
    """
    # Load original MR metadata
    mr_metadata, target_shape = load_original_mr(mr_path)
    
    # Load synthetic CT
    synthetic_ct, sample_name = load_synthetic_ct(npz_path)
    
    # Upscale synthetic CT
    upscaled_ct = upscale_with_sitk(synthetic_ct, mr_metadata, target_shape, interpolation)
    
    # Verify shape
    expected_shape = (mr_metadata['size'][2], mr_metadata['size'][1], mr_metadata['size'][0])
    if upscaled_ct.shape != expected_shape:
        print(f"⚠️ Warning: Output shape {upscaled_ct.shape} doesn't match expected {expected_shape}")
    
    # Save result
    output_file = save_upscaled_ct(upscaled_ct, sample_name, mr_metadata, output_dir, save_format)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Upscale synthetic CT using original MR metadata')
    parser.add_argument('--npz', '-n', type=str, required=True,
                       help='NPZ file containing synthetic CT')
    parser.add_argument('--mr', '-m', type=str, required=True,
                       help='Original MR file (.mha, .nii, .nii.gz)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--format', '-f', type=str, default='mha',
                       choices=['mha', 'nii', 'nii.gz'],
                       help='Output format (default: mha)')
    parser.add_argument('--interpolation', '-i', type=str, default='linear',
                       choices=['linear', 'nearest', 'bspline'],
                       help='Interpolation method (default: linear)')
    
    args = parser.parse_args()
    
    print("🚀 Simple Synthetic CT Upscaling")
    print(f"   NPZ file: {args.npz}")
    print(f"   MR file: {args.mr}")
    print(f"   Output: {args.output}")
    print(f"   Format: {args.format}")
    print(f"   Interpolation: {args.interpolation}")
    print("-" * 50)
    
    try:
        output_file = process_file(
            npz_path=args.npz,
            mr_path=args.mr,
            output_dir=args.output,
            save_format=args.format,
            interpolation=args.interpolation
        )
        
        print(f"\n✅ Success!")
        print(f"📁 Upscaled CT saved: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
