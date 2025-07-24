#!/usr/bin/env python3
"""
Modified test script that saves inference results as NPZ files instead of images
Includes CT denormalization and optional real CT replacement using file paths.
"""

import time
import os
import numpy as np
import SimpleITK as sitk
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

# CT Denormalization parameters (matching CT_normalization.ipynb)
HU_MIN = -1024  # Air
HU_MAX = 3000   # Dense soft tissue/bone

def denormalize_ct(normalized_ct, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Convert normalized [-1,1] CT values back to Hounsfield Units
    """
    return (normalized_ct + 1) * (hu_max - hu_min) / 2 + hu_min

def is_ct_data(data, key_name):
    if 'B' in key_name and hasattr(data, 'min') and hasattr(data, 'max'):
        data_min, data_max = data.min(), data.max()
        return data_min >= -1.1 and data_max <= 1.1
    return False

opt = TestOptions().parse()
opt.nThreads = 0
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

# Optional: path to real CT directory
real_ct_root = None
use_external_real_ct = real_ct_root is not None

# Add this to TestOptions manually if not present:
# self.parser.add_argument('--real_ct_root', type=str, default=None, help='Optional path to ground truth CTs')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

npz_output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}_npz')
os.makedirs(npz_output_dir, exist_ok=True)

print(f"Saving NPZ files to: {npz_output_dir}")
dataset_size = len(dataset)
total_samples = min(opt.how_many, dataset_size)
print(f"Dataset size: {dataset_size}")
print(f"Processing {total_samples} samples")

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    print(f"Processing sample {i+1}/{total_samples}")

    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()

    if isinstance(img_path, list):
        base_name = os.path.splitext(os.path.basename(img_path[0]))[0]
    else:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Replace real_B with external CT if available
    if use_external_real_ct:
        real_ct_path = real_ct_root
        if os.path.exists(real_ct_path):
            real_ct_itk = sitk.ReadImage(real_ct_path)
            real_ct_np = sitk.GetArrayFromImage(real_ct_itk)
            visuals['real_B'] = real_ct_np
            print(f"  Replaced real_B with external CT: {real_ct_path}")
        else:
            print(f"  ⚠️ Warning: Could not find real CT file at {real_ct_path}")

    # Save tensors as NPZ with CT denormalization
    npz_data = {}
    for key, val in visuals.items():
        if hasattr(val, 'cpu'):
            numpy_data = val.cpu().numpy().squeeze()
        else:
            numpy_data = val.squeeze() if hasattr(val, 'squeeze') else val

        if is_ct_data(numpy_data, key):
            original_range = f"[{numpy_data.min():.3f}, {numpy_data.max():.3f}]"
            numpy_data = denormalize_ct(numpy_data)
            hu_range = f"[{numpy_data.min():.1f}, {numpy_data.max():.1f}] HU"
            print(f"  {key}: {numpy_data.shape} - Denormalized {original_range} → {hu_range}")
        else:
            print(f"  {key}: {numpy_data.shape} - Kept original values")

        npz_data[key] = numpy_data

    npz_filename = os.path.join(npz_output_dir, f"{base_name}_{i:04d}.npz")
    np.savez_compressed(npz_filename, **npz_data)
    print(f"  Saved: {npz_filename}")

print(f"\nCompleted! Saved {total_samples} NPZ files to: {npz_output_dir}")
print("Each NPZ file contains:")
print("  - 'real_A': Input MR data")
print("  - 'fake_B': Generated synthetic CT (denormalized to HU)")
print("  - 'real_B': Ground truth CT (external if provided)")
print("\n✅ CT Denormalization:")
print(f"  [-1,1] → [{HU_MIN}, {HU_MAX}] HU")
print("  Matches normalization from CT_normalization.ipynb")