"""
4-Channel Test Dataset Loader for 3D Medical Imaging
Loads 4-channel MR latent data for inference and includes original MR dimensions
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader


def find_original_mr_file(sample_name, dataroot):
    """
    Find the original MR image file for a given sample
    
    Args:
        sample_name: Base sample name (e.g., '1HNA010')
        dataroot: Root directory to search for MR files
    
    Returns:
        Path to image file or None if not found
    """
    # Common patterns for MR file naming with various extensions
    possible_patterns = [
        f"{sample_name}.mha",
        f"{sample_name}_mr.mha", 
        f"{sample_name}_MR.mha",
        f"{sample_name}_t1.mha",
        f"{sample_name}_T1.mha",
        f"{sample_name}.nii",
        f"{sample_name}.nii.gz",
        f"{sample_name}_mr.nii",
        f"{sample_name}_MR.nii",
        f"{sample_name}_t1.nii",
        f"{sample_name}_T1.nii",
        f"{sample_name}_mr.nii.gz",
        f"{sample_name}_MR.nii.gz",
        f"{sample_name}_t1.nii.gz",
        f"{sample_name}_T1.nii.gz"
    ]
    
    # Search in common directories
    search_dirs = [
        dataroot,
        os.path.join(dataroot, ".."),
        os.path.join(dataroot, "..", "MR"),
        os.path.join(dataroot, "..", "mr"),
        os.path.join(dataroot, "..", "original"),
        os.path.join(dataroot, "..", "..", "MR"),
        os.path.join(dataroot, "..", "..", "original"),
        os.path.join(dataroot, "..", "..", "raw"),
        # Add more potential paths based on your data structure
        os.path.dirname(dataroot),  # Parent directory
        os.path.join(os.path.dirname(dataroot), "MR"),
        os.path.join(os.path.dirname(dataroot), "original")
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for pattern in possible_patterns:
            file_path = os.path.join(search_dir, pattern)
            if os.path.exists(file_path):
                return file_path
        
        # Also search recursively in subdirectories
        try:
            for root, dirs, files in os.walk(search_dir):
                for pattern in possible_patterns:
                    if pattern in files:
                        return os.path.join(root, pattern)
        except:
            continue
    
    return None

def get_target_shape_and_spacing_from_mha(sample_name, dataroot):
    """
    Get target dimensions and spacing by loading the original MR .mha file
    
    Args:
        sample_name: Base sample name
        dataroot: Root directory to search for MR files
    
    Returns:
        tuple: (target_shape, spacing, origin) or (None, None, None) if file not found
        - target_shape: numpy array with shape [depth, height, width]
        - spacing: tuple with spacing [z_spacing, y_spacing, x_spacing]
        - origin: tuple with origin [z_origin, y_origin, x_origin]
    """
    mr_file_path = find_original_mr_file(sample_name, dataroot)
    
    if mr_file_path is None:
        print(f"⚠️ Warning: Could not find original MR file for {sample_name}")
        return None, None, None

    
    # Load the MR image using SimpleITK
    mr_image = sitk.ReadImage(mr_file_path)
    size = mr_image.GetSize()  # Returns (width, height, depth)
    spacing = mr_image.GetSpacing()  # Returns (x_spacing, y_spacing, z_spacing)
    origin = mr_image.GetOrigin()  # Returns (x_origin, y_origin, z_origin)
    
    # Convert to (depth, height, width) format
    target_shape = np.array([size[2], size[1], size[0]])
    
    # Convert spacing to (z_spacing, y_spacing, x_spacing) to match target_shape order
    spacing_dhw = (spacing[2], spacing[1], spacing[0])
    origin_dhw = (origin[2], origin[1], origin[0])
         
    print(f"📏 Found original MR for {sample_name}: {mr_file_path}")
    print(f"   Target shape: {target_shape}")
    print(f"   Spacing (D,H,W): {spacing_dhw}")
    print(f"   Origin (D,H,W): {origin_dhw}")

    return target_shape, spacing_dhw, origin_dhw

def get_target_shape_from_mha(sample_name, dataroot):
    """
    Backward compatibility function - returns only target shape
    """
    target_shape, _, _ = get_target_shape_and_spacing_from_mha(sample_name, dataroot)
    return target_shape

class FourChannelTestDataset(Dataset):
    """
    Test dataset class for 4-channel MR latent inference
    
    Loads latent MR data and includes original MR dimensions for upscaling
    """
    
    def __init__(self, opt):
        """
        Initialize 4-channel test dataset
        
        Args:
            opt: Options containing dataset configuration
                - dataroot: Root directory (should be latent_scaled folder or test data folder)
                - phase: 'test'
        """
        self.opt = opt
        self.root = opt.dataroot
        
        # Set up folder paths for MR test data
        self.mr_dir = os.path.join(self.root, "MR")
        
        # If MR subfolder doesn't exist, assume dataroot contains the npz files directly
        if not os.path.exists(self.mr_dir):
            self.mr_dir = self.root
        
        # Validate directory exists
        if not os.path.exists(self.mr_dir):
            raise FileNotFoundError(f"MR directory not found: {self.mr_dir}")
        
        # Find all NPZ files in MR directory
        mr_files = [f for f in os.listdir(self.mr_dir) if f.endswith('.npz')]
        
        if len(mr_files) == 0:
            raise ValueError(f"No NPZ files found in {self.mr_dir}")
        
        # Create list of valid samples
        self.samples = []
        
        print(f"🔍 Found {len(mr_files)} MR files in test directory")
        
        for mr_file in mr_files:
            mr_path = os.path.join(self.mr_dir, mr_file)
            
            # Extract sample name (remove latent_mr.npz suffix if present)
            sample_name = mr_file.replace('_latent_mr.npz', '').replace('_mr.npz', '').replace('.npz', '')
            
            # Load to check if it's valid 4-channel data
            try:
                data = np.load(mr_path)
                
                # Check if it has the expected keys and shape
                if 'latent_mr' in data:
                    latent_mr = data['latent_mr']
                elif 'data' in data:
                    latent_mr = data['data']
                else:
                    # Assume the first key contains the data
                    latent_mr = data[list(data.keys())[0]]
                
                # Validate shape (should be 4-channel: (4, D, H, W))
                if len(latent_mr.shape) == 4 and latent_mr.shape[0] == 4:
                    # Get target shape, spacing, and origin from original MR .mha file
                    target_shape, spacing, origin = get_target_shape_and_spacing_from_mha(sample_name, self.root)
                    
                    # Fallback to stored shape if MR file not found
                    if target_shape is None:
                        if 'target_shape' in data:
                            target_shape = data['target_shape']
                        elif 'original_shape' in data:
                            target_shape = data['original_shape']
                        else:
                            # Default fallback: scale up by common factor
                            target_shape = np.array([latent_mr.shape[1] * 4, latent_mr.shape[2], latent_mr.shape[3]])
                            print(f"⚠️ Using default target shape for {sample_name}: {target_shape}")
                        
                        # Default spacing and origin if not available
                        spacing = (1.0, 1.0, 1.0)  # Default 1mm isotropic spacing
                        origin = (0.0, 0.0, 0.0)   # Default origin
                        print(f"⚠️ Using default spacing and origin for {sample_name}")
                    
                    # Get original MR data if available in NPZ, otherwise load from .mha file
                    if 'original_mr' in data:
                        original_mr = data['original_mr']
                    else:
                        # Load original MR data from .mha file
                        mr_file_path = find_original_mr_file(sample_name, self.root)
                        if mr_file_path is not None:
                            try:
                                mr_image = sitk.ReadImage(mr_file_path)
                                original_mr = sitk.GetArrayFromImage(mr_image)  # Returns numpy array
                                print(f"📂 Loaded original MR data for {sample_name}: {original_mr.shape}")
                            except Exception as e:
                                print(f"❌ Error loading original MR data for {sample_name}: {e}")
                                original_mr = np.array([])  # Empty placeholder as fallback
                        else:
                            original_mr = np.array([])  # Empty placeholder if file not found
                    
                    self.samples.append({
                        'mr_path': mr_path,
                        'sample_name': sample_name,
                        'target_shape': target_shape,
                        'spacing': spacing,
                        'origin': origin,
                        'original_mr': original_mr
                    })
                    
                    print(f"✅ Valid: {sample_name} - Shape: {latent_mr.shape} → Target: {target_shape} (Spacing: {spacing})")
                else:
                    print(f"❌ Invalid shape for {sample_name}: {latent_mr.shape} (expected 4-channel)")
                    
            except Exception as e:
                print(f"❌ Error loading {mr_file}: {e}")
                continue
        
        if len(self.samples) == 0:
            raise ValueError("No valid 4-channel samples found")
        
        print(f"✅ 4-Channel test dataset initialized:")
        print(f"   Root: {self.root}")
        print(f"   Phase: test")
        print(f"   Samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a test sample
        
        Returns:
            dict with keys:
                'A': 4-channel MR latent tensor [4, D, H, W]
                'sample_name': string identifier
                'target_shape': target dimensions for upscaling
                'spacing': spacing from original MR (z, y, x)
                'origin': origin from original MR (z, y, x)
                'original_mr': original MR data (if available)
        """
        sample = self.samples[index]
        
        # Load MR latent data
        data = np.load(sample['mr_path'])
        
        # Get latent MR data
        if 'latent_mr' in data:
            latent_mr = data['latent_mr']
        elif 'data' in data:
            latent_mr = data['data']
        else:
            latent_mr = data[list(data.keys())[0]]
        
        # Ensure it's float32 and normalized to [-1, 1] if needed
        latent_mr = latent_mr.astype(np.float32)
        
        # Normalize if values are outside [-1, 1] range
        if latent_mr.max() > 1.0 or latent_mr.min() < -1.0:
            # Assume it's in [0, 1] range and convert to [-1, 1]
            if latent_mr.min() >= 0:
                latent_mr = latent_mr * 2.0 - 1.0
            else:
                # Already in some other range, normalize to [-1, 1]
                latent_mr = 2.0 * (latent_mr - latent_mr.min()) / (latent_mr.max() - latent_mr.min()) - 1.0
        
        # Convert to torch tensor
        latent_mr_tensor = torch.from_numpy(latent_mr)
        
        return {
            'A': latent_mr_tensor,
            'sample_name': sample['sample_name'],
            'target_shape': torch.from_numpy(np.array(sample['target_shape'])),
            'spacing': sample['spacing'],
            'origin': sample['origin'],
            'original_mr': sample['original_mr']
        }


class FourChannelTestDataLoaderWrapper:
    """
    Wrapper class to provide data loader interface for test dataset
    """
    
    def __init__(self, opt):
        self.opt = opt
        self.dataset = FourChannelTestDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize if hasattr(opt, 'batchSize') else 1,
            shuffle=False,  # Don't shuffle for test
            num_workers=1,  # Single worker for test
            drop_last=False  # Keep all samples
        )
    
    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataset)
