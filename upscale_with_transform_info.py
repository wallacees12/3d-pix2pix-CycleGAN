#!/usr/bin/env python3
"""
Upscale synthetic CT using downsampling information from crops.pkl
Reverses the original MR → latent transformation
"""

import os
import pickle
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
from scipy.ndimage import zoom

def load_downsampling_info(crops_pkl_path, sample_name):
    """
    Load downsampling info for a specific sample from crops.pkl
    """
    with open(crops_pkl_path, 'rb') as f:
        crops_data = pickle.load(f)
    
    # Find entry for this sample
    for entry in crops_data:
        if entry['name'] == sample_name:
            if 'bounds' in entry and entry['bounds'] is not None:
                return entry['bounds'].get('downsampling_info')
            break
    
    return None

def upscale_with_downsampling_info(synthetic_ct, downsampling_info, method='scipy'):
    """
    Upscale synthetic CT using original downsampling transformation info
    """
    if downsampling_info is None:
        print("⚠️  No downsampling info available, using 4x default scale")
        scale_factors = (4.0, 4.0, 4.0)
        target_shape = tuple(int(s * scale) for s, scale in zip(synthetic_ct.shape, scale_factors))
    else:
        # Use exact inverse transformation
        scale_factors = downsampling_info['scale_factors']
        target_shape = downsampling_info['original_shape']
        
        print(f"📐 Using original transformation:")
        print(f"   Original shape: {target_shape}")
        print(f"   Latent shape: {downsampling_info['latent_shape']}")
        print(f"   Scale factors: {scale_factors}")
        print(f"   Original spacing: {downsampling_info['original_spacing']}")
    
    # Perform upscaling
    if method == 'sitk' and downsampling_info is not None:
        # Use SimpleITK with proper spacing
        sitk_img = sitk.GetImageFromArray(synthetic_ct)
        sitk_img.SetSpacing(downsampling_info['latent_spacing'])
        sitk_img.SetOrigin(downsampling_info['original_origin'])
        sitk_img.SetDirection(downsampling_info['original_direction'])
        
        # Resample to original space
        resample = sitk.ResampleImageFilter()
        resample.SetSize([target_shape[2], target_shape[1], target_shape[0]])  # x,y,z order
        resample.SetOutputSpacing(downsampling_info['original_spacing'])
        resample.SetOutputOrigin(downsampling_info['original_origin'])
        resample.SetOutputDirection(downsampling_info['original_direction'])
        resample.SetInterpolator(sitk.sitkLinear)
        
        upscaled_img = resample.Execute(sitk_img)
        upscaled_ct = sitk.GetArrayFromImage(upscaled_img)
    else:
        # Use scipy zoom (same as training pipeline)
        upscaled_ct = zoom(synthetic_ct, scale_factors, order=1)
    
    print(f"🔄 Upscaled: {synthetic_ct.shape} → {upscaled_ct.shape}")
    return upscaled_ct

def main():
    parser = argparse.ArgumentParser(description='Upscale synthetic CT using downsampling information')
    parser.add_argument('--npz_file', required=True, help='NPZ file with synthetic CT')
    parser.add_argument('--crops_pkl', required=True, help='crops.pkl file with downsampling info')
    parser.add_argument('--method', choices=['scipy', 'sitk'], default='scipy', 
                       help='Upscaling method (default: scipy)')
    parser.add_argument('--output_dir', help='Output directory (default: same as input)')
    
    args = parser.parse_args()
    
    # Load synthetic CT
    npz_path = Path(args.npz_file)
    data = np.load(npz_path)
    
    if 'synthetic_ct' in data:
        synthetic_ct = data['synthetic_ct']
    elif 'data' in data:
        synthetic_ct = data['data']
    elif 'fake_B' in data:
        synthetic_ct = data['fake_B']
    else:
        print(f"❌ No synthetic CT data found in {npz_path}")
        print(f"Available keys: {list(data.keys())}")
        return 1
    
    # Extract sample name from filename
    sample_name = npz_path.stem.replace('_synthetic_ct', '').replace('_result', '').replace('_fake_B', '')
    print(f"🔍 Processing sample: {sample_name}")
    print(f"📊 Synthetic CT shape: {synthetic_ct.shape}")
    
    # Load downsampling info
    downsampling_info = load_downsampling_info(args.crops_pkl, sample_name)
    
    if downsampling_info:
        print(f"✅ Found downsampling info for {sample_name}")
    else:
        print(f"⚠️  No downsampling info found for {sample_name}")
    
    # Upscale
    upscaled_ct = upscale_with_downsampling_info(synthetic_ct, downsampling_info, args.method)
    
    # Save result
    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    output_path = output_dir / f"{sample_name}_upscaled.npz"
    
    np.savez_compressed(output_path, data=upscaled_ct, synthetic_ct=upscaled_ct)
    print(f"💾 Saved: {output_path}")
    
    # Also save as MHA if we have spacing info
    if downsampling_info:
        mha_path = output_dir / f"{sample_name}_upscaled.mha"
        sitk_img = sitk.GetImageFromArray(upscaled_ct)
        sitk_img.SetSpacing(downsampling_info['original_spacing'])
        sitk_img.SetOrigin(downsampling_info['original_origin'])
        sitk_img.SetDirection(downsampling_info['original_direction'])
        sitk.WriteImage(sitk_img, str(mha_path))
        print(f"💾 Saved with proper spacing: {mha_path}")

if __name__ == "__main__":
    exit(main())
