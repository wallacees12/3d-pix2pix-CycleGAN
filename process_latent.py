#!/usr/bin/env python3
"""
Upscaling Script for Synthetic CT with Advanced Resampling
Upscales low-resolution synthetic CT to full resolution using proper medical imaging resampling
Handles spacing information and provides multiple interpolation methods
"""

import argparse
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from pathlib import Path

def load_npz_data(npz_file):
    """Load data from NPZ file with target_shape from test script"""
    data = np.load(npz_file)
    
    # Check what keys are available
    available_keys = list(data.keys())
    print(f"Available keys in NPZ: {available_keys}")
    
    # Load synthetic CT data (fake_B)
    if 'fake_B' in data:
        ct_data = data['fake_B']
        print(f"📊 Found synthetic CT data (fake_B)")
    else:
        raise ValueError("No 'fake_B' (synthetic CT) data found in NPZ file")
    
    print(f"   Shape: {ct_data.shape}")
    print(f"   HU range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
    
    # Load target dimensions from test script
    target_shape = None
    if 'target_shape' in data:
        target_shape = tuple(data['target_shape'])
        print(f"📐 Found target dimensions from original MR: {target_shape}")
    else:
        print("⚠️  No target_shape found in NPZ")
    
    # Load additional info for reference
    sample_name = None
    if 'sample_name' in data:
        sample_name = str(data['sample_name'])
        print(f"👤 Sample name: {sample_name}")
    
    original_mr_shape = None
    if 'original_mr_shape' in data:
        original_mr_shape = tuple(data['original_mr_shape'])
        print(f"📏 Original MR shape: {original_mr_shape}")
    
    latent_shape = None
    if 'latent_shape' in data:
        latent_shape = tuple(data['latent_shape'])
        print(f"🔬 Latent shape: {latent_shape}")
    
    # For reference - show real_A if available
    if 'real_A' in data:
        real_A = data['real_A']
        print(f"🔍 Reference MR (real_A) shape: {real_A.shape}")
    
    return ct_data, target_shape, sample_name

def load_reconstruction_metadata(sample_name, dataset_root=None):
    """
    Load reconstruction metadata from crops.pkl for proper upscaling
    
    Args:
        sample_name: Patient/sample name
        dataset_root: Path to dataset root containing crops.pkl
    
    Returns:
        crop_info dict with spacing and zoom factor information or None
    """
    if dataset_root is None:
        return None
        
    import pickle
    crops_file = Path(dataset_root) / "crops.pkl"
    
    if not crops_file.exists():
        print(f"⚠️  No crops.pkl found at {crops_file}")
        return None
    
    try:
        with open(crops_file, 'rb') as f:
            dataset_entries = pickle.load(f)
        
        # Find entry for this sample
        for entry in dataset_entries:
            if entry.get('name') == sample_name:
                crop_info = entry.get('bounds', {})
                if crop_info:
                    print(f"📋 Found reconstruction metadata for {sample_name}:")
                    print(f"   Original shape: {crop_info.get('original_shape')}")
                    print(f"   Target shape: {crop_info.get('target_shape')}")
                    print(f"   Zoom factors: {crop_info.get('zoom_factors')}")
                    print(f"   Original spacing: {crop_info.get('spacing_original')}")
                    print(f"   Resampled spacing: {crop_info.get('spacing_resampled')}")
                    return crop_info
                else:
                    print(f"ℹ️  No reconstruction metadata for {sample_name}")
                    return None
        
        print(f"⚠️  Sample {sample_name} not found in crops.pkl")
        return None
        
    except Exception as e:
        print(f"❌ Error loading reconstruction metadata: {e}")
        return None

def upscale_ct_scipy(ct_data, target_shape=None, scale_factor=None, order=1):
    """
    Upscale CT data using scipy.ndimage.zoom (same method as downscaling)
    
    Args:
        ct_data: Input CT data [D, H, W]
        target_shape: Target shape (D, H, W) or None
        scale_factor: Scale factor (float) or None
        order: Interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic)
    
    Returns:
        Upscaled CT data
    """
    original_shape = ct_data.shape
    print(f"🔄 Original shape: {original_shape}")
    
    if target_shape is not None:
        # Calculate zoom factors for each dimension
        zoom_factors = [target_shape[i] / original_shape[i] for i in range(3)]
        print(f"📏 Target shape: {target_shape}")
    elif scale_factor is not None:
        # Use uniform scale factor
        zoom_factors = [scale_factor, scale_factor, scale_factor]
        target_shape = tuple(int(original_shape[i] * scale_factor) for i in range(3))
        print(f"📈 Scale factor: {scale_factor}")
        print(f"📏 Target shape: {target_shape}")
    else:
        # Default: upscale by 4x (common for 32→128 or 128→512)
        scale_factor = 4.0
        zoom_factors = [scale_factor, scale_factor, scale_factor]
        target_shape = tuple(int(original_shape[i] * scale_factor) for i in range(3))
        print(f"📈 Default scale factor: {scale_factor}")
        print(f"📏 Target shape: {target_shape}")
    
    print(f"🔍 Zoom factors: {zoom_factors}")
    
    # Perform interpolation using scipy (same as downscaling process)
    print(f"🚀 Starting scipy upscaling (order={order})...")
    upscaled_ct = zoom(ct_data, zoom_factors, order=order, prefilter=False)
    
    print(f"✅ Upscaling completed!")
    print(f"📊 Upscaled shape: {upscaled_ct.shape}")
    print(f"   HU range: [{upscaled_ct.min():.1f}, {upscaled_ct.max():.1f}]")
    
    return upscaled_ct

def upscale_ct_sitk(ct_data, target_shape, crop_info=None):
    """
    Upscale CT data using SimpleITK with proper spacing handling
    Uses reconstruction metadata from training for accurate upscaling
    
    Args:
        ct_data: Input CT data [D, H, W]
        target_shape: Target shape (D, H, W)
        crop_info: Reconstruction metadata dict from crops.pkl
    
    Returns:
        Upscaled CT data
    """
    original_shape = ct_data.shape
    print(f"🔄 SimpleITK resampling with reconstruction metadata:")
    print(f"   Original shape: {original_shape}")
    print(f"   Target shape: {target_shape}")
    
    # Create SimpleITK image from numpy array
    sitk_img = sitk.GetImageFromArray(ct_data)
    
    # Use reconstruction metadata if available
    if crop_info:
        print("📋 Using reconstruction metadata from training:")
        
        # Get spacing information from crop_info
        spacing_original = crop_info.get('spacing_original', [1.0, 1.0, 1.0])
        spacing_resampled = crop_info.get('spacing_resampled', [1.0, 1.0, 1.0])
        zoom_factors = crop_info.get('zoom_factors', [1.0, 1.0, 1.0])
        
        print(f"   Training zoom factors: {zoom_factors}")
        print(f"   Original spacing: {spacing_original}")
        print(f"   Resampled spacing: {spacing_resampled}")
        
        # Set the latent image spacing (what the model was trained on)
        sitk_img.SetSpacing(spacing_resampled)
        
        # Calculate target spacing to reconstruct original resolution
        # Inverse of the zoom factors applied during training
        target_spacing = spacing_original
        
    else:
        print("⚠️  No reconstruction metadata, using estimated spacing:")
        
        # Calculate spacing based on zoom factors (fallback)
        zoom_factors = [target_shape[i] / original_shape[i] for i in range(3)]
        original_spacing = [1.0, 1.0, 1.0]  # Default
        target_spacing = [original_spacing[i] / zoom_factors[i] for i in range(3)]
        
        sitk_img.SetSpacing(original_spacing)
    
    sitk_img.SetOrigin([0.0, 0.0, 0.0])
    
    print(f"🔍 Upscaling spacing:")
    print(f"   Input spacing: {sitk_img.GetSpacing()}")
    print(f"   Target spacing: {target_spacing}")
    
    # Create reference image with target properties
    reference_img = sitk.Image(target_shape[::-1], sitk.sitkFloat32)  # ITK order is (W, H, D)
    reference_img.SetSpacing(target_spacing)
    reference_img.SetOrigin([0.0, 0.0, 0.0])
    
    # Resample using high-quality interpolation
    print("🚀 Starting SimpleITK resampling...")
    resampled_img = sitk.Resample(sitk_img, reference_img, 
                                  sitk.Transform(), 
                                  sitk.sitkLinear,  # Can be changed to sitkBSpline for higher quality
                                  0.0,  # Default pixel value
                                  sitk.sitkFloat32)
    
    # Convert back to numpy array
    upscaled_ct = sitk.GetArrayFromImage(resampled_img)
    
    print(f"✅ SimpleITK resampling completed!")
    print(f"📊 Upscaled shape: {upscaled_ct.shape}")
    print(f"   HU range: [{upscaled_ct.min():.1f}, {upscaled_ct.max():.1f}]")
    
    return upscaled_ct

def save_upscaled_data(upscaled_ct, original_file, sample_name=None, output_format='npz'):
    """
    Save upscaled CT data
    
    Args:
        upscaled_ct: Upscaled CT data
        original_file: Path to original NPZ file
        sample_name: Sample name for output file
        output_format: 'npz', 'mha', or 'both'
    """
    original_path = Path(original_file)
    output_dir = original_path.parent
    
    # Determine base name
    if sample_name:
        base_name = f"{sample_name}_upscaled"
    else:
        base_name = original_path.stem.replace('_synthetic_ct', '_upscaled')
    
    output_files = []
    
    if output_format in ['npz', 'both']:
        # Save as NPZ
        output_file = output_dir / f"{base_name}.npz"
        np.savez_compressed(output_file, 
                           data=upscaled_ct,
                           fake_B=upscaled_ct,
                           shape=np.array(upscaled_ct.shape),
                           sample_name=sample_name or "unknown")
        print(f"💾 Saved NPZ: {output_file}")
        output_files.append(output_file)
        
    if output_format in ['mha', 'both']:
        # Save as MHA using SimpleITK
        output_file = output_dir / f"{base_name}.mha"
        
        # Create SimpleITK image
        sitk_img = sitk.GetImageFromArray(upscaled_ct)
        
        # Set spacing to 1.0 (can be adjusted if needed)
        sitk_img.SetSpacing([1.0, 1.0, 1.0])
        sitk_img.SetOrigin([0.0, 0.0, 0.0])
        
        # Write MHA file
        sitk.WriteImage(sitk_img, str(output_file))
        print(f"💾 Saved MHA: {output_file}")
        output_files.append(output_file)
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description='Upscale synthetic CT using advanced resampling methods')
    parser.add_argument('--file', required=True, help='Path to NPZ file from test_4channel_simple.py')
    parser.add_argument('--dataset_root', help='Path to dataset root containing crops.pkl for reconstruction metadata')
    parser.add_argument('--scale', type=float, help='Scale factor (overrides target_shape from NPZ)')
    parser.add_argument('--target_shape', nargs=3, type=int, help='Manual target shape D H W (overrides NPZ)')
    parser.add_argument('--format', choices=['npz', 'mha', 'both'], default='both', 
                       help='Output format (default: both)')
    parser.add_argument('--method', choices=['scipy', 'sitk'], default='scipy',
                       help='Upscaling method: scipy (same as downscaling) or sitk (medical imaging resampling)')
    parser.add_argument('--order', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Interpolation order for scipy: 0=nearest, 1=linear, 2=quadratic, 3=cubic (default: 1)')
    parser.add_argument('--ignore_target', action='store_true',
                       help='Ignore target_shape from NPZ and use scale factor instead')
    parser.add_argument('--ignore_metadata', action='store_true',
                       help='Ignore reconstruction metadata from crops.pkl')
    parser.add_argument('--clip_hu', action='store_true',
                       help='Clip HU values to valid range [-1024, 3071]')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return 1
    
    if not args.file.endswith('.npz'):
        print(f"❌ Input must be an NPZ file, got: {args.file}")
        return 1
    
    print(f"🔧 Upscaling Configuration:")
    print(f"   Input file: {args.file}")
    print(f"   Dataset root: {args.dataset_root}")
    print(f"   Manual scale factor: {args.scale}")
    print(f"   Manual target shape: {args.target_shape}")
    print(f"   Output format: {args.format}")
    print(f"   Resampling method: {args.method}")
    print(f"   Interpolation order: {args.order}")
    print(f"   Ignore NPZ target: {args.ignore_target}")
    print(f"   Ignore metadata: {args.ignore_metadata}")
    print(f"   Clip HU values: {args.clip_hu}")
    print()
    
    try:
        # Load CT data and target dimensions
        print("📂 Loading NPZ file from test script...")
        ct_data, npz_target_shape, sample_name = load_npz_data(args.file)
        
        # Load reconstruction metadata if available
        crop_info = None
        if not args.ignore_metadata and sample_name and args.dataset_root:
            print(f"📋 Loading reconstruction metadata for {sample_name}...")
            crop_info = load_reconstruction_metadata(sample_name, args.dataset_root)
        elif not args.dataset_root:
            print("ℹ️  No dataset_root provided, skipping reconstruction metadata")
        
        # Determine target shape priority:
        # 1. Manual target_shape (if provided)
        # 2. Manual scale factor (if provided)
        # 3. Target shape from NPZ (if available and not ignored)
        # 4. Target shape from reconstruction metadata
        # 5. Default 4x scale
        
        target_shape = None
        scale_factor = None
        
        if args.target_shape:
            target_shape = tuple(args.target_shape)
            print(f"🎯 Using manual target shape: {target_shape}")
        elif args.scale:
            scale_factor = args.scale
            print(f"🎯 Using manual scale factor: {scale_factor}")
        elif npz_target_shape and not args.ignore_target:
            target_shape = npz_target_shape
            print(f"🎯 Using target shape from NPZ: {target_shape}")
        elif crop_info and crop_info.get('original_shape'):
            target_shape = tuple(crop_info['original_shape'])
            print(f"🎯 Using target shape from reconstruction metadata: {target_shape}")
        else:
            scale_factor = 4.0
            print(f"🎯 Using default scale factor: {scale_factor}")
        
        # Choose upscaling method
        if args.method == 'scipy':
            print(f"🔬 Using scipy resampling (same method as downscaling process)")
            upscaled_ct = upscale_ct_scipy(ct_data, target_shape=target_shape, 
                                          scale_factor=scale_factor, order=args.order)
        elif args.method == 'sitk':
            print(f"🏥 Using SimpleITK medical imaging resampling")
            if target_shape is None:
                # Calculate target shape from scale factor
                original_shape = ct_data.shape
                target_shape = tuple(int(original_shape[i] * scale_factor) for i in range(3))
            upscaled_ct = upscale_ct_sitk(ct_data, target_shape, crop_info=crop_info)
        
        # Clip HU values to valid range if requested
        if args.clip_hu:
            print("🔧 Clipping HU values to valid range [-1024, 3071]...")
            original_min, original_max = upscaled_ct.min(), upscaled_ct.max()
            upscaled_ct = np.clip(upscaled_ct, -1024, 3071)
            clipped_min, clipped_max = upscaled_ct.min(), upscaled_ct.max()
            if original_min < -1024 or original_max > 3071:
                print(f"   Clipped from [{original_min:.1f}, {original_max:.1f}] to [{clipped_min:.1f}, {clipped_max:.1f}] HU")
            else:
                print(f"   No clipping needed: [{clipped_min:.1f}, {clipped_max:.1f}] HU")
        
        # Save upscaled data
        print("💾 Saving upscaled data...")
        output_files = save_upscaled_data(upscaled_ct, args.file, sample_name, args.format)
        
        # Calculate scaling information
        original_shape = ct_data.shape
        actual_scale_factors = [upscaled_ct.shape[i] / original_shape[i] for i in range(3)]
        
        print(f"\n✅ Upscaling completed successfully!")
        print(f"📁 Input: {args.file}")
        print(f"📁 Output files:")
        for output_file in output_files:
            print(f"   - {output_file}")
        print(f"📊 Shape transformation: {original_shape} → {upscaled_ct.shape}")
        print(f"� Actual scale factors: {[f'{f:.2f}x' for f in actual_scale_factors]}")
        print(f"🔬 Method used: {args.method}")
        
        if npz_target_shape:
            print(f"📐 NPZ target shape: {npz_target_shape}")
            if target_shape == npz_target_shape:
                print("✅ Perfect match with original MR dimensions!")
            else:
                print("ℹ️  Different from original MR dimensions (manual override)")
        
        if crop_info:
            original_shape_from_metadata = crop_info.get('original_shape')
            if original_shape_from_metadata and target_shape == tuple(original_shape_from_metadata):
                print("✅ Perfect reconstruction to original training data dimensions!")
            else:
                print("ℹ️  Different from original training data dimensions")
        
        if sample_name:
            print(f"👤 Sample: {sample_name}")
        
        # Give recommendations
        print(f"\n💡 Recommendations:")
        if args.method == 'scipy':
            print(f"   - Using scipy method (same as training data processing)")
            print(f"   - For medical accuracy, consider --method sitk")
        elif args.method == 'sitk':
            print(f"   - Using medical imaging resampling for better anatomical accuracy")
            if crop_info:
                print(f"   - Using reconstruction metadata for accurate spacing")
        
        if not args.clip_hu:
            print(f"   - Consider --clip_hu to ensure valid HU range")
        
        if not args.dataset_root and sample_name:
            print(f"   - Consider --dataset_root for reconstruction metadata")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
