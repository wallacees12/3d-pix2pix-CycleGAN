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

def extract_downsampling_info(original_path, latent_path, ct_path=None):
    """
    Extract downsampling/transformation info from original image to latent MR
    
    Args:
        original_path: Path to original high-res MR (.mha)
        latent_path: Path to latent representation (.npz)
        ct_path: Path to original CT (.mha) - preferred for spacing info if available
    
    Returns:
        dict: Transformation metadata for upscaling
    """
    # Load original MR for shape information
    original_img = sitk.ReadImage(original_path)
    original_data = sitk.GetArrayFromImage(original_img)
    
    # Use CT for spacing/orientation if available, otherwise use MR
    if ct_path and os.path.exists(ct_path):
        print(f"   📊 Using CT spacing info from {os.path.basename(ct_path)}")
        reference_img = sitk.ReadImage(ct_path)
        reference_data = sitk.GetArrayFromImage(reference_img)
        # Use CT spacing but ensure shape matches MR (they should be registered)
        if reference_data.shape != original_data.shape:
            print(f"   ⚠️  CT shape {reference_data.shape} != MR shape {original_data.shape}, using MR")
            reference_img = original_img
            reference_data = original_data
    else:
        print(f"   📊 Using MR spacing info from {os.path.basename(original_path)}")
        reference_img = original_img
        reference_data = original_data
    
    reference_spacing = reference_img.GetSpacing()
    reference_origin = reference_img.GetOrigin()
    reference_direction = reference_img.GetDirection()
    
    # Load latent MR
    latent_data = np.load(latent_path)['data']  # Shape: (D,H,W,4) or (4,D,H,W)
    
    # Handle different latent formats
    if latent_data.shape[-1] == 4:  # (D,H,W,4)
        latent_shape = latent_data.shape[:3]
    elif latent_data.shape[0] == 4:  # (4,D,H,W)
        latent_shape = latent_data.shape[1:]
    else:
        raise ValueError(f"Unexpected latent shape: {latent_data.shape}")
    
    # Calculate transformation parameters using reference image
    scale_factors = tuple(ref/lat for ref, lat in zip(reference_data.shape, latent_shape))
    new_spacing = tuple(ref_sp * scale for ref_sp, scale in zip(reference_spacing, scale_factors))
    
    return {
        'original_shape': reference_data.shape,  # Target shape for upscaling
        'latent_shape': latent_shape,
        'original_spacing': reference_spacing,   # Target spacing (from CT if available)
        'latent_spacing': new_spacing,
        'scale_factors': scale_factors,
        'original_origin': reference_origin,
        'original_direction': reference_direction,
        'reference_type': 'CT' if ct_path and os.path.exists(ct_path) else 'MR'
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
        
        # Look for corresponding original MR file and CT file
        possible_original_paths = [
            data_dir / f"{patient_id}_mr.mha"
        ]
        
        possible_ct_paths = [
            data_dir / f"{patient_id}_ct.mha",
            data_dir / f"{patient_id}.mha",
            data_dir / f"{patient_id}_CT.mha"
        ]
        
        original_path = None
        for path in possible_original_paths:
            if path.exists():
                original_path = path
                break
        
        ct_path = None
        for path in possible_ct_paths:
            if path.exists():
                ct_path = path
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
                downsampling_info = extract_downsampling_info(original_path, latent_file, ct_path)
                crop_entry['bounds'] = {
                    'downsampling_info': downsampling_info,
                    'original_mr_path': original_path.name,
                    'original_ct_path': ct_path.name if ct_path else None
                }
                ref_type = downsampling_info.get('reference_type', 'MR')
                print(f"✅ {patient_id}: Added downsampling info from {ref_type}")
                print(f"   Scale factors: {downsampling_info['scale_factors']}")
                if ct_path:
                    print(f"   Using CT spacing: {ct_path.name}")
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
        
        original_files = list(data_dir.glob("*_mr.mha"))
        print(f"\nFound original MR files:")
        for f in original_files:
            print(f"  📁 {f.name}")
        
        ct_files = list(data_dir.glob("*_ct.mha")) or list(data_dir.glob("*_CT.mha"))
        print(f"\nFound original CT files:")
        for f in ct_files:
            print(f"  🏥 {f.name}")
        
        if ct_files:
            print("\n💡 CT files found! These will be used for spacing information (preferred over MR)")
        else:
            print("\n💡 No CT files found, will use MR spacing information")
    else:
        create_test_crops(args.data_dir, args.output)

if __name__ == "__main__":
    main()
