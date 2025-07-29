#!/usr/bin/env python3
"""
Normalize Dataset CT Files
Converts a folder of CT.mha files to NPZ format with proper normalization
Based on Normalisation.ipynb but uses [-1024, 3000] HU windowing
"""

import numpy as np
import SimpleITK as sitk
import argparse
import os
from pathlib import Path
from scipy import ndimage
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def normalize_ct_file(ct_path, target_shape=(32, 128, 128), hu_min=-1024, hu_max=1200, output_dir=None):
    """
    Normalize a single CT file following the Normalisation.ipynb workflow
    
    Args:
        ct_path: Path to input CT.mha file
        target_shape: Target dimensions (D, H, W)
        hu_min: Minimum HU value for windowing
        hu_max: Maximum HU value for windowing  
        output_dir: Output directory (default: same as input)
    
    Returns:
        dict: Processing metadata
    """
    ct_path = Path(ct_path)
    if output_dir is None:
        output_dir = ct_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🏥 Processing: {ct_path.name}")
    
    # 1. Load CT data using SimpleITK
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")
    
    ct_image = sitk.ReadImage(str(ct_path))
    ct_array = sitk.GetArrayFromImage(ct_image)
    
    original_shape = ct_array.shape
    original_spacing = ct_image.GetSpacing()
    original_origin = ct_image.GetOrigin()
    original_direction = ct_image.GetDirection()
    
    # print(f"   📊 Original shape: {original_shape}")
    # print(f"   📏 Original spacing: {original_spacing}")
    # print(f"   📈 HU range: [{ct_array.min():.1f}, {ct_array.max():.1f}]")
    
    # 2. Calculate zoom factors for spatial resampling
    zoom_factors = [
        target_shape[0] / original_shape[0],  # depth
        target_shape[1] / original_shape[1],  # height  
        target_shape[2] / original_shape[2]   # width
    ]
    
    # print(f"   🔍 Zoom factors: {zoom_factors}")
    # print(f"   🎯 Target shape: {target_shape}")
    
    # 3. Perform spatial resampling using linear interpolation
    # print(f"   🔄 Resampling...")
    ct_resampled = ndimage.zoom(ct_array, zoom_factors, order=1)
    
    if ct_resampled.shape != target_shape:
        print(f"   ⚠️  Shape mismatch! Got {ct_resampled.shape}, expected {target_shape}")
    
    # 4. Apply intensity normalization
    print(f"   🎨 Normalizing HU range [{hu_min}, {hu_max}] → [-1, 1]...")
    
    # Clip extreme values
    ct_clipped = np.clip(ct_resampled, hu_min, hu_max)
    
    # Normalize to [-1, 1] range
    ct_normalized = (ct_clipped - hu_min) / (hu_max - hu_min)  # Scale to [0, 1]
    ct_normalized = ct_normalized * 2 - 1  # Scale to [-1, 1]
    
    print(f"   ✅ Normalized range: [{ct_normalized.min():.3f}, {ct_normalized.max():.3f}]")
    
    # 5. Calculate new spacing after resampling
    new_spacing = [
        original_spacing[0] * original_shape[2] / target_shape[2],  # x-spacing
        original_spacing[1] * original_shape[1] / target_shape[1],  # y-spacing
        original_spacing[2] * original_shape[0] / target_shape[0]   # z-spacing
    ]
    
    # 6. Save as NPZ file
    parent_name = ct_path.parent.name
    output_name = f"{parent_name}_latent_ct.npz"
    output_path = output_dir / output_name
    
    # Convert to float32 for consistency
    ct_final = ct_normalized.astype(np.float32)
    
    # Save NPZ with comprehensive metadata
    np.savez_compressed(
        output_path,
        data=ct_final,
        ct=ct_final,  # Alternative key for compatibility
        original_shape=np.array(original_shape),
        target_shape=np.array(target_shape),
        zoom_factors=np.array(zoom_factors),
        original_spacing=np.array(original_spacing),
        new_spacing=np.array(new_spacing),
        original_origin=np.array(original_origin),
        original_direction=np.array(original_direction),
        hu_range=np.array([hu_min, hu_max]),
        processing_date=str(datetime.now().isoformat())
    )
    
    print(f"   💾 Saved: {output_name}")
    
    # 7. Create processing metadata
    metadata = {
        'original_shape': original_shape,
        'target_shape': target_shape,
        'zoom_factors': zoom_factors,
        'spacing_original': original_spacing,
        'spacing_resampled': new_spacing,
        'hu_range': (hu_min, hu_max),
        'processing_date': datetime.now().isoformat(),
        'input_file': str(ct_path),
        'output_file': str(output_path)
    }
    
    return metadata

def process_ct_folder(input_dir, output_dir=None, target_shape=(32, 128, 128), 
                     hu_min=-1024, hu_max=1400, save_crops_pkl=True):
    """
    Process all CT.mha files in a folder
    
    Args:
        input_dir: Directory containing CT.mha files
        output_dir: Output directory (default: same as input)
        target_shape: Target dimensions (D, H, W)
        hu_min: Minimum HU value for windowing
        hu_max: Maximum HU value for windowing
        save_crops_pkl: Whether to save crops.pkl file
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Scanning {input_dir} for CT.mha files...")
    
    # Find all CT.mha files
    ct_files = list(input_dir.rglob("ct.mha"))
    ct_files = [f for f in ct_files if f.is_file()]
    
    if not ct_files:
        print(f"❌ No .mha files found in {input_dir}")
        return
    
    print(f"📁 Found {len(ct_files)} CT files")
    
    # Process each file
    all_metadata = []
    successful_files = []
    failed_files = []
    
    for i, ct_file in enumerate(ct_files, 1):
        print(f"\n[{i}/{len(ct_files)}] Processing {ct_file.name}...")
        
        try:
            metadata = normalize_ct_file(
                ct_path=ct_file,
                target_shape=target_shape,
                hu_min=hu_min,
                hu_max=hu_max,
                output_dir=output_dir
            )
            all_metadata.append(metadata)
            successful_files.append(ct_file)
            
        except Exception as e:
            print(f"   ❌ Error processing {ct_file.name}: {e}")
            failed_files.append((ct_file, str(e)))
            continue
    
    # Save crops.pkl file for compatibility with training pipeline
    if save_crops_pkl and successful_files:
        print(f"\n📋 Creating crops.pkl file...")
        crops_data = []
        
        for ct_file, metadata in zip(successful_files, all_metadata):
            # Extract patient ID from filename
            patient_id = ct_file.stem.replace('_ct', '').replace('_CT', '').replace('_normalized', '')
            
            # Create crop entry compatible with training pipeline
            crop_entry = {
                'name': patient_id,
                'mr_path': None,  # No MR for CT-only processing
                'ct_path': ct_file.name.replace('.mha', '_normalized.npz'),
                'phase': 'train',  # Default phase
                'bounds': {
                    'original_shape': {'ct': metadata['original_shape']},
                    'target_shape': metadata['target_shape'],
                    'zoom_factors': {'ct': metadata['zoom_factors']},
                    'spacing_original': {'ct': metadata['spacing_original']},
                    'spacing_resampled': {'ct': metadata['spacing_resampled']},
                    'hu_range': metadata['hu_range'],
                    'processing_date': metadata['processing_date']
                }
            }
            crops_data.append(crop_entry)
        
        crops_pkl_path = output_dir / "crops.pkl"
        with open(crops_pkl_path, 'wb') as f:
            pickle.dump(crops_data, f)
        
        print(f"   💾 Saved crops.pkl with {len(crops_data)} entries")
    
    # Print summary
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"=" * 50)
    print(f"✅ Successfully processed: {len(successful_files)}/{len(ct_files)} files")
    print(f"❌ Failed: {len(failed_files)} files")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for failed_file, error in failed_files:
            print(f"   {failed_file.name}: {error}")
    
    print(f"\n📊 Processing settings:")
    print(f"   Target shape: {target_shape}")
    print(f"   HU window: [{hu_min}, {hu_max}]")
    print(f"   Output range: [-1, 1]")
    print(f"   Output directory: {output_dir}")
    
    if successful_files:
        # Show sample file info
        sample_npz = output_dir / (successful_files[0].stem + "_normalized.npz")
        if sample_npz.exists():
            sample_data = np.load(sample_npz)
            print(f"\n📋 Sample output (first file):")
            print(f"   NPZ keys: {list(sample_data.keys())}")
            print(f"   Data shape: {sample_data['data'].shape}")
            print(f"   Data range: [{sample_data['data'].min():.3f}, {sample_data['data'].max():.3f}]")

def main():
    parser = argparse.ArgumentParser(description='Normalize CT dataset using extended HU window')
    parser.add_argument('input_dir', help='Directory containing CT.mha files')
    parser.add_argument('--output_dir', help='Output directory (default: same as input)')
    parser.add_argument('--target_shape', nargs=3, type=int, default=[32, 128, 128],
                       help='Target shape (D H W) (default: 32 128 128)')
    parser.add_argument('--hu_min', type=int, default=-1024,
                       help='Minimum HU value for windowing (default: -1024)')
    parser.add_argument('--hu_max', type=int, default=1400,
                       help='Maximum HU value for windowing (default: 3000)')
    parser.add_argument('--no_crops_pkl', action='store_true',
                       help='Skip creating crops.pkl file')
    parser.add_argument('--list_files', action='store_true',
                       help='List files without processing')
    
    args = parser.parse_args()
    
    if args.list_files:
        input_dir = Path(args.input_dir)
        ct_files = list(input_dir.rglob("ct.mha"))
        ct_files = [f for f in ct_files if f.is_file()]
        
        print(f"🔍 Found {len(ct_files)} CT files in {input_dir}:")
        for ct_file in ct_files:
            print(f"   📄 {ct_file.relative_to(input_dir)}")
        return
    
    # Validate input
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1
    
    print(f"🏥 CT Dataset Normalization")
    print(f"=" * 50)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir or args.input_dir}")
    print(f"🎯 Target shape: {tuple(args.target_shape)}")
    print(f"🪟 HU window: [{args.hu_min}, {args.hu_max}]")
    print(f"📊 Output range: [-1, 1]")
    print()
    
    # Process files
    process_ct_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_shape=tuple(args.target_shape),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        save_crops_pkl=not args.no_crops_pkl
    )

if __name__ == "__main__":
    exit(main())