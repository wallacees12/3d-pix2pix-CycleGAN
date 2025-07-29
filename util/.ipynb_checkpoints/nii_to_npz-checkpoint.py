#!/usr/bin/env python3
"""
Convert Latent .nii files to NPZ format for 4-channel dataset training
Supports global or local normalization to [-1, 1]
"""

import os
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import json


def compute_global_stats_percentile(nii_files, lower_pct=0.5, upper_pct=99.5):
    """
    Approximate global intensity percentiles across many files by streaming.
    
    For each file, we compute the lower and upper percentiles and keep the tightest
    global window that includes all of them. This avoids high memory usage.
    
    Returns:
        dict with 'min' and 'max' keys (estimated percentile bounds).
    """
    import numpy as np
    import SimpleITK as sitk

    global_lower = float('inf')
    global_upper = float('-inf')

    print(f"🧮 Streaming estimate of global {lower_pct}-{upper_pct} percentiles from {len(nii_files)} files...")

    for i, nii_file in enumerate(nii_files):
        try:
            array = sitk.GetArrayFromImage(sitk.ReadImage(str(nii_file))).flatten()
            local_lower = np.percentile(array, lower_pct)
            local_upper = np.percentile(array, upper_pct)
            
            global_lower = min(global_lower, local_lower)
            global_upper = max(global_upper, local_upper)

            print(f"   [{i+1}/{len(nii_files)}] {nii_file.name}: {lower_pct:.1f}%={local_lower:.3f}, {upper_pct:.1f}%={local_upper:.3f}")
        
        except Exception as e:
            print(f"   ⚠️ Error reading {nii_file}: {e}")
    
    print(f"\n✅ Final global percentile window: [{global_lower:.3f}, {global_upper:.3f}]")
    return {'min': float(global_lower), 'max': float(global_upper)}


def convert_nii_to_npz(nii_file, output_dir, data_type='mr', channels=4,
                       global_stats=None, normalization='global'):
    """
    Convert a single .nii file to NPZ format
    """
    print(f"🔄 Converting {nii_file}...")
    
    # Load NII file using SimpleITK
    sitk_image = sitk.ReadImage(nii_file)
    array = sitk.GetArrayFromImage(sitk_image)
    
    print(f"   Original shape: {array.shape}")
    
    if data_type == 'mr':
        # Shape: (D, H, W, C) or (C, D, H, W)
        if len(array.shape) == 4:
            if array.shape[-1] == channels:
                array = np.transpose(array, (3, 0, 1, 2))
                print(f"   Transposed to: {array.shape} (channel-first)")
            elif array.shape[0] == channels:
                print(f"   Already channel-first: {array.shape}")
            else:
                raise ValueError(f"Unexpected MR shape: {array.shape}")
        elif len(array.shape) == 3:
            print(f"   ⚠️  3D MR data without channels, expanding to {channels} channels")
            array = np.repeat(array[np.newaxis, ...], channels, axis=0)
        else:
            raise ValueError(f"Unexpected MR dimensionality: {array.shape}")
        
    elif data_type == 'ct':
        if len(array.shape) == 4 and (array.shape[0] == 1 or array.shape[-1] == 1):
            array = array.squeeze()
            print(f"   Squeezed CT to: {array.shape}")
        elif len(array.shape) == 3:
            print(f"   Preserving 3D CT volume: {array.shape}")
        else:
            raise ValueError(f"Unexpected CT shape: {array.shape}")
    
    # Normalize
    print(f"   Original value range: [{array.min():.3f}, {array.max():.3f}]")
    
    if normalization == 'global' and global_stats:
        gmin, gmax = global_stats['min'], global_stats['max']
        print(f"   📊 Global normalization using min={gmin:.3f}, max={gmax:.3f}")
        array = np.clip(array, gmin, gmax)
        array = (array - gmin) / (gmax - gmin)
        array = array * 2.0 - 1.0
    else:
        local_min = array.min()
        local_max = array.max()
        print(f"   📊 Local normalization using min={local_min:.3f}, max={local_max:.3f}")
        if local_max > local_min:
            array = (array - local_min) / (local_max - local_min)
            array = array * 2.0 - 1.0
        else:
            print(f"   ⚠️ Constant image, skipping normalization")
    
    print(f"   Normalized range: [{array.min():.3f}, {array.max():.3f}]")
    
    # Save
    input_name = Path(nii_file).stem
    output_file = os.path.join(output_dir, f"{input_name}.npz")
    np.savez_compressed(output_file, data=array)
    
    print(f"   ✅ Saved: {output_file}")
    return output_file


def batch_convert_dataset(input_dir, output_base_dir, mr_pattern="*latent_mr.nii*", ct_pattern="*ct*.nii*",
                          normalization='global', global_stats=None):
    """
    Batch convert MR and CT files
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    mr_output_dir = output_path / "MR"
    ct_output_dir = output_path / "CT"
    mr_output_dir.mkdir(parents=True, exist_ok=True)
    ct_output_dir.mkdir(parents=True, exist_ok=True)
    
    mr_files = list(input_path.glob(mr_pattern))
    ct_files = list(input_path.glob(ct_pattern))
    
    print(f"\n📁 Found MR: {len(mr_files)}, CT: {len(ct_files)}")
    
    # Compute global stats if needed
    if normalization == 'global' and global_stats is None:
        global_stats = compute_global_stats_percentile(mr_files + ct_files)
        with open(output_path / "global_stats.json", 'w') as f:
            json.dump(global_stats, f)
    
    # Convert MR
    for f in mr_files:
        try:
            convert_nii_to_npz(f, mr_output_dir, data_type='mr', channels=4,
                               global_stats=global_stats, normalization=normalization)
        except Exception as e:
            print(f"❌ Error MR {f}: {e}")
    
    # Convert CT
    for f in ct_files:
        try:
            convert_nii_to_npz(f, ct_output_dir, data_type='ct', channels=1,
                               global_stats=global_stats, normalization=normalization)
        except Exception as e:
            print(f"❌ Error CT {f}: {e}")
    
    return global_stats


def main():
    parser = argparse.ArgumentParser(description='Convert .nii to .npz with normalization')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mr_pattern', type=str, default='*latent_mr.nii*')
    parser.add_argument('--ct_pattern', type=str, default='*ct*.nii*')
    parser.add_argument('--normalization', choices=['global', 'local'], default='global')
    parser.add_argument('--global_stats', type=str, help='Path to global_stats.json if precomputed')
    parser.add_argument('--dataset_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"🔄 Converting NII to NPZ with {args.normalization} normalization")
    global_stats = None
    
    if args.global_stats:
        print(f"📖 Loading global stats from {args.global_stats}")
        with open(args.global_stats, 'r') as f:
            global_stats = json.load(f)
    
    batch_convert_dataset(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        mr_pattern=args.mr_pattern,
        ct_pattern=args.ct_pattern,
        normalization=args.normalization,
        global_stats=global_stats
    )
    
    print(f"\n✅ Conversion complete. NPZ files saved to {args.output_dir}")


if __name__ == '__main__':
    main()