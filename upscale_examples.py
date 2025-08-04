#!/usr/bin/env python3
"""
Example usage of the simple upscale_synthetic_ct.py script
Shows how to upscale synthetic CT using original MR files directly
"""

import os

def run_upscaling_examples():
    """Show different ways to use the simple upscaling script"""
    
    print("🎯 Simple Synthetic CT Upscaling Examples")
    print("=" * 50)
    
    # Example paths (adjust these to your actual paths)
    npz_file = "./results/experiment_name/test_latest_npz/1HNA006_synthetic_ct.npz"
    mr_file = "/home/sawall/scratch/latent_data/latent_scaled/HN/test/MR/1HNA006_mr.mha"
    output_dir = "./upscaled_results/"
    
    print("\n1️⃣ Basic upscaling (recommended):")
    cmd1 = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir}"
    print(f"   {cmd1}")
    
    print("\n2️⃣ Different output formats:")
    print("   # MHA format (default, best for medical software)")
    cmd2a = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir} -f mha"
    print(f"   {cmd2a}")
    
    print("   # NIfTI format (common in neuroimaging)")
    cmd2b = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir} -f nii.gz"
    print(f"   {cmd2b}")
    
    print("\n3️⃣ Different interpolation methods:")
    print("   # Linear interpolation (default, good balance)")
    cmd3a = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir} -i linear"
    print(f"   {cmd3a}")
    
    print("   # Nearest neighbor (preserves exact values)")
    cmd3b = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir} -i nearest"
    print(f"   {cmd3b}")
    
    print("   # B-spline (smooth, high quality)")
    cmd3c = f"python upscale_synthetic_ct.py -n {npz_file} -m {mr_file} -o {output_dir} -i bspline"
    print(f"   {cmd3c}")
    
    print("\n🔧 Key Features:")
    print("   ✅ Uses original MR file directly for all metadata")
    print("   ✅ Perfect spacing and origin preservation")
    print("   ✅ SimpleITK medical-grade resampling")
    print("   ✅ Simple command-line interface")
    print("   ✅ Robust error handling")
    
    print("\n📊 Interpolation Methods:")
    print("   • linear:   Good balance of speed and quality (default)")
    print("   • nearest:  Fastest, preserves exact HU values")  
    print("   • bspline:  Highest quality, smooth results")
    
    print("\n🎯 Typical workflow:")
    print("   1. Generate NPZ: python test_4channel.py --save_type npz")
    print("   2. Upscale:      python upscale_synthetic_ct.py -n file.npz -m original.mha -o ./out/")
    print("   3. View:         Load .mha file in 3D Slicer or medical software")

def show_batch_processing():
    """Show how to process multiple files"""
    
    print("\n" + "=" * 50)
    print("🔄 Batch Processing Examples")
    print("=" * 50)
    
    print("\nFor batch processing, you can use shell scripts:")
    
    batch_script = '''#!/bin/bash
# Batch upscaling script

NPZ_DIR="./results/experiment_name/test_latest_npz"
MR_DIR="/home/sawall/scratch/latent_data/latent_scaled/HN/test/MR"
OUTPUT_DIR="./upscaled_results"

# Process each NPZ file
for npz_file in "$NPZ_DIR"/*_synthetic_ct.npz; do
    # Extract sample name (e.g., 1HNA006 from 1HNA006_synthetic_ct.npz)
    sample_name=$(basename "$npz_file" _synthetic_ct.npz)
    
    # Find corresponding MR file
    mr_file="$MR_DIR/${sample_name}_mr.mha"
    
    if [ -f "$mr_file" ]; then
        echo "Processing $sample_name..."
        python upscale_synthetic_ct.py -n "$npz_file" -m "$mr_file" -o "$OUTPUT_DIR"
    else
        echo "Warning: MR file not found for $sample_name"
    fi
done
'''
    
    print("Bash script (save as 'batch_upscale.sh'):")
    print(batch_script)
    
    python_script = '''#!/usr/bin/env python3
import os
import subprocess
import glob

npz_dir = "./results/experiment_name/test_latest_npz"
mr_dir = "/home/sawall/scratch/latent_data/latent_scaled/HN/test/MR"
output_dir = "./upscaled_results"

# Find all NPZ files
npz_files = glob.glob(os.path.join(npz_dir, "*_synthetic_ct.npz"))

for npz_file in npz_files:
    # Extract sample name
    sample_name = os.path.basename(npz_file).replace("_synthetic_ct.npz", "")
    
    # Find corresponding MR file
    mr_file = os.path.join(mr_dir, f"{sample_name}_mr.mha")
    
    if os.path.exists(mr_file):
        print(f"Processing {sample_name}...")
        cmd = [
            "python", "upscale_synthetic_ct.py",
            "-n", npz_file,
            "-m", mr_file,
            "-o", output_dir
        ]
        subprocess.run(cmd)
    else:
        print(f"Warning: MR file not found for {sample_name}")
'''
    
    print("\nPython script (save as 'batch_upscale.py'):")
    print(python_script)

def validate_requirements():
    """Show how to validate files and requirements"""
    
    print("\n" + "=" * 50)
    print("🔍 File Validation")
    print("=" * 50)
    
    validation_code = '''import numpy as np
import SimpleITK as sitk
import os

def validate_npz_file(npz_path):
    """Check if NPZ file has required synthetic CT data"""
    try:
        data = np.load(npz_path)
        
        # Check for synthetic CT data
        ct_keys = ['fake_B', 'synthetic_ct']
        ct_data = None
        for key in ct_keys:
            if key in data and data[key].ndim == 3:
                ct_data = data[key]
                break
        
        if ct_data is None:
            return False, "No 3D synthetic CT data found"
        
        print(f"✅ NPZ file valid: {npz_path}")
        print(f"   CT shape: {ct_data.shape}")
        print(f"   HU range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

def validate_mr_file(mr_path):
    """Check if MR file can be loaded and has metadata"""
    try:
        image = sitk.ReadImage(mr_path)
        
        print(f"✅ MR file valid: {mr_path}")
        print(f"   Size: {image.GetSize()} (x,y,z)")
        print(f"   Spacing: {image.GetSpacing()} (x,y,z)")
        print(f"   Origin: {image.GetOrigin()} (x,y,z)")
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

# Example usage
npz_file = "path/to/synthetic_ct.npz"
mr_file = "path/to/original_mr.mha"

npz_valid, npz_msg = validate_npz_file(npz_file)
mr_valid, mr_msg = validate_mr_file(mr_file)

if npz_valid and mr_valid:
    print("🎯 Ready for upscaling!")
else:
    print("❌ Fix issues before upscaling")
'''
    
    print("Python validation script:")
    print(validation_code)

if __name__ == "__main__":
    run_upscaling_examples()
    show_batch_processing()
    validate_requirements()
    
    print("\n" + "=" * 50)
    print("🚀 Ready to upscale with simple, reliable workflow!")
    print("=" * 50)
