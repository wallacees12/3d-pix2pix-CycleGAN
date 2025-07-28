#!/usr/bin/env python3
"""
Generate crops.pkl for 4-channel latent MR testing
Works with existing launch_4channel_training.py pipeline
"""

import os
import pickle
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse

def extract_downsampling_info(original_path, latent_path):
    """
    Extract downsampling/transformation info from original MR to latent MR
    
    Args:
        original_path: Path to original high-res MR (.mha)
        latent_path: Path to latent representation (.npz)
    
    Returns:
        dict: Transformation metadata for upscaling
    """
    # Load original MR
    original_img = sitk.ReadImage(original_path)
    original_data = sitk.GetArrayFromImage(original_img)
    original_spacing = original_img.GetSpacing()
    original_origin = original_img.GetOrigin()
    
    # Load latent MR
    latent_data = np.load(latent_path)['data']  # Shape: (D,H,W,4) or (4,D,H,W)
    
    # Handle different latent formats
    if latent_data.shape[-1] == 4:  # (D,H,W,4)
        latent_shape = latent_data.shape[:3]
    elif latent_data.shape[0] == 4:  # (4,D,H,W)
        latent_shape = latent_data.shape[1:]
    else:
        raise ValueError(f"Unexpected latent shape: {latent_data.shape}")
    
    # Calculate transformation parameters
    scale_factors = tuple(orig/lat for orig, lat in zip(original_data.shape, latent_shape))
    new_spacing = tuple(orig_sp * scale for orig_sp, scale in zip(original_spacing, scale_factors))
    
    return {
        'original_shape': original_data.shape,
        'latent_shape': latent_shape,
        'original_spacing': original_spacing,
        'latent_spacing': new_spacing,
        'scale_factors': scale_factors,
        'original_origin': original_origin,
        'original_direction': original_img.GetDirection()
    }

def create_test_crops(data_dir, output_path=None):
    """
    Create crops.pkl for testing with 4-channel latent MR data
    
    Args:
        data_dir: Directory containing test data
        output_path: Output path for crops.pkl (default: data_dir/crops.pkl)
    """
    data_dir = Path(data_dir)
    
    if output_path is None:
        output_path = data_dir / "crops.pkl"
    
    # Find latent MR files
    latent_files = list(data_dir.glob("*_mr_all_channels.npz"))
    
    print(f"🔍 Found {len(latent_files)} latent MR files")
    
    crops_data = []
    
    for latent_file in sorted(latent_files):
        # Extract patient ID from filename
        patient_id = latent_file.stem.replace('_mr_all_channels', '').replace('_all_channels', '')
        
        # Look for corresponding original MR file
        possible_original_paths = [
            data_dir / f"{patient_id}_mr.mha"
        ]
        
        original_path = None
        for path in possible_original_paths:
            if path.exists():
                original_path = path
                break
        
        # Create crop entry
        crop_entry = {
            'name': patient_id,
            'mr_path': latent_file.name,  # Latent MR for inference
            'ct_path': None,  # No CT for testing
            'phase': 'test'
        }
        
        # Add downsampling info if original MR found
        if original_path:
            try:
                downsampling_info = extract_downsampling_info(original_path, latent_file)
                crop_entry['bounds'] = {
                    'downsampling_info': downsampling_info,
                    'original_mr_path': original_path.name
                }
                print(f"✅ {patient_id}: Added downsampling info from {original_path.name}")
                print(f"   Scale factors: {downsampling_info['scale_factors']}")
            except Exception as e:
                print(f"⚠️  {patient_id}: Could not extract downsampling info: {e}")
                crop_entry['bounds'] = None
        else:
            print(f"⚠️  {patient_id}: No original MR found, upscaling will use default scale")
            crop_entry['bounds'] = None
        
        crops_data.append(crop_entry)
    
    # Save crops.pkl
    with open(output_path, 'wb') as f:
        pickle.dump(crops_data, f)
    
    print(f"\n✅ Created {output_path}")
    print(f"📊 {len(crops_data)} test samples ready")
    print(f"💡 Compatible with launch_4channel_training.py --phase test")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create crops.pkl for 4-channel latent MR testing')
    parser.add_argument('data_dir', help='Directory containing latent MR files and original MR files')
    parser.add_argument('--output', help='Output path for crops.pkl (default: data_dir/crops.pkl)')
    parser.add_argument('--list-files', action='store_true', help='List found files without creating crops.pkl')
    
    args = parser.parse_args()
    
    if args.list_files:
        # Just list what we would process
        data_dir = Path(args.data_dir)
        print(f"🔍 Scanning {data_dir} for test files...")
        
        latent_files = list(data_dir.glob("*_latent.npz")) or list(data_dir.glob("*.npz"))
        print(f"\nFound latent files:")
        for f in latent_files:
            try:
                data = np.load(f)
                if 'data' in data:
                    print(f"  ✅ {f.name}: shape {data['data'].shape}")
            except:
                print(f"  ❌ {f.name}: could not load")
        
        original_files = list(data_dir.glob("*.mha"))
        print(f"\nFound original MR files:")
        for f in original_files:
            print(f"  📁 {f.name}")
    else:
        create_test_crops(args.data_dir, args.output)

if __name__ == "__main__":
    main()
