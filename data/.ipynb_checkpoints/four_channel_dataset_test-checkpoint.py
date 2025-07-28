"""
4-Channel Test Dataset Loader for 3D Medical Imaging
Specifically designed for inference-only testing with 4-channel MR data
Loads original MR files to get target dimensions for upscaling
"""

import os
import numpy as np
import pickle
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

class FourChannelTestDataset(Dataset):
    """
    Test dataset class for 4-channel MR inference with original MR for target dimensions
    
    Expected crops.pkl format:
    [
        {
            'name': 'patient_id',  # e.g., '1HNA001' 
            'mr_path': 'patient_mr_all_channels.npz',  # 4-channel latent MR
        },
        ...
    ]
    
    Expected files in dataset directory:
    - {patient_id}_mr_all_channels.npz  # 4-channel latent MR for inference
    - {patient_id}.mha                  # Original full-resolution MR for target dimensions
    """
    
    def __init__(self, opt):
        """
        Initialize 4-channel test dataset
        
        Args:
            opt: Options containing dataset configuration
                - dataroot: Root directory containing the dataset
                - phase: Should be 'test'
                - input_nc: Number of input channels (should be 4)
                - output_nc: Number of output channels (should be 1)
        """
        self.opt = opt
        self.root = opt.dataroot
        
        # Load dataset metadata
        crops_file = os.path.join(self.root, "crops.pkl")
        if not os.path.exists(crops_file):
            raise FileNotFoundError(f"Dataset metadata not found: {crops_file}")
        
        with open(crops_file, 'rb') as f:
            self.samples = pickle.load(f)
        
        print(f"📊 Loaded {len(self.samples)} test samples from {crops_file}")
        
        # Validate sample data (simplified for test mode)
        for i, sample in enumerate(self.samples):
            required_keys = ['name', 'mr_path']
            for key in required_keys:
                if key not in sample:
                    raise KeyError(f"Sample {i} missing required key: {key}")
            
            # Check if original MR file exists
            patient_id = sample['name']
            original_mr_path = os.path.join(self.root, f"{patient_id}_mr.mha")
            if not os.path.exists(original_mr_path):
                print(f"⚠️  Original MR not found for {patient_id}: {original_mr_path}")
                print(f"    Will use scale estimation for target dimensions")
        
        print(f"✅ 4-Channel test dataset initialized:")
        print(f"   Root: {self.root}")
        print(f"   Phase: test (inference only)")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Expected: 4-channel latent MR + original MR files")
    
    def name(self):
        """Return dataset name"""
        return 'FourChannelTestDataset'
    
    def initialize(self, opt):
        """Initialize method for compatibility with base dataset interface"""
        # Dataset is already initialized in __init__, this is for compatibility
        pass
    
    def __len__(self):
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a test data sample with both latent MR and original MR for target dimensions
        
        Returns:
            dict containing:
                - A: 4-channel MR latent tensor [4, D_low, H_low, W_low]
                - B: dummy 1-channel tensor [1, D_low, H_low, W_low] (zeros)
                - A_paths: MR latent file path
                - B_paths: empty string (no CT in test mode)
                - sample_name: patient/sample name for result saving
                - original_mr: original full-resolution MR for target dimensions
                - target_shape: target dimensions for upscaling
        """
        sample = self.samples[index]
        patient_id = sample['name']
        
        # Load MR latent data (4-channel, low resolution)
        mr_latent_path = os.path.join(self.root, sample['mr_path'])
        if not os.path.exists(mr_latent_path):
            raise FileNotFoundError(f"MR latent file not found: {mr_latent_path}")
        
        # Load NPZ file with 4-channel MR latent data
        mr_latent_data = np.load(mr_latent_path, allow_pickle=True)['data']  # Shape: [D, H, W, C]
        
        # Validate and process MR latent data
        if len(mr_latent_data.shape) == 4:
            # Transpose from [D, H, W, C] to [C, D, H, W]
            mr_latent_data = np.transpose(mr_latent_data, (3, 0, 1, 2))
            if mr_latent_data.shape[0] != 4:
                raise ValueError(f"Expected 4 MR channels, got {mr_latent_data.shape[0]} in {sample['mr_path']}")
        else:
            raise ValueError(f"Invalid MR data shape {mr_latent_data.shape} in {sample['mr_path']}. Expected [D, H, W, 4]")
        
        # Load original full-resolution MR for target dimensions
        original_mr_data = None
        target_shape = None
        
        # Look for original MR file with pattern {patient_id}.mha
        original_mr_path = os.path.join(self.root, f"{patient_id}_mr.mha")
        
        if os.path.exists(original_mr_path):
            try:
                # Load original MR with nibabel
                
                img = sitk.ReadImage(original_mr_path)
                original_mr_data = sitk.GetArrayFromImage(img)
                
                # Get target shape (spatial dimensions only)
                if len(original_mr_data.shape) == 4:
                    target_shape = original_mr_data.shape[:3]  # [D, H, W]
                elif len(original_mr_data.shape) == 3:
                    target_shape = original_mr_data.shape      # [D, H, W]
                
                print(f"📐 Loaded original MR for {patient_id}: {original_mr_path}")
                print(f"   Original MR shape: {original_mr_data.shape}")
                print(f"   Target dimensions: {target_shape}")
                
            except Exception as e:
                print(f"⚠️  Could not load original MR from {original_mr_path}: {e}")
                original_mr_data = None
                target_shape = None
        else:
            print(f"⚠️  Original MR not found: {original_mr_path}")
        
        # If no original MR found, estimate target shape (e.g., 4x upscale)
        if target_shape is None:
            scale_factor = 4  # Common scale factor for medical imaging
            target_shape = tuple(int(mr_latent_data.shape[i+1] * scale_factor) for i in range(3))
            print(f"📏 Estimated target dimensions (4x scale): {target_shape}")
            # Create dummy original MR data with target shape
            original_mr_data = np.zeros(target_shape)
        
        # Convert MR latent to tensor
        mr_tensor = torch.from_numpy(mr_latent_data.astype(np.float32))
        
        # Create dummy CT tensor (not used in inference, but needed for model compatibility)
        ct_tensor = torch.zeros(1, *mr_latent_data.shape[1:], dtype=torch.float32)
        
        # Apply normalization if specified
        if hasattr(self.opt, 'normalize_input') and self.opt.normalize_input:
            mr_tensor = self._normalize_tensor(mr_tensor)
        
        return {
            'A': mr_tensor,                    # 4-channel MR latent
            'B': ct_tensor,                    # Dummy CT tensor (zeros)
            'A_paths': mr_latent_path,         # MR latent file path
            'B_paths': "",                     # Empty string (no CT)
            'sample_name': patient_id,         # Patient/sample name for saving results
            'original_mr': original_mr_data,   # Original MR for reference (can be zeros if not found)
            'target_shape': np.array(target_shape)  # Target dimensions for upscaling
        }
    
    def _normalize_tensor(self, tensor):
        """
        Normalize tensor to [-1, 1] range
        
        Args:
            tensor: Input tensor
        
        Returns:
            Normalized tensor
        """
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max > tensor_min:
            # Normalize to [0, 1] then to [-1, 1]
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
            tensor = tensor * 2.0 - 1.0
        
        return tensor
    
    def get_sample_info(self, index):
        """Get information about a specific test sample"""
        sample = self.samples[index]
        patient_id = sample['name']
        
        # Load data to get shapes
        mr_path = os.path.join(self.root, sample['mr_path'])
        mr_data = np.load(mr_path, allow_pickle=True)['data']
        
        # Check for original MR
        original_mr_path = os.path.join(self.root, f"{patient_id}.mha")
        original_mr_exists = os.path.exists(original_mr_path)
        
        info = {
            'name': patient_id,
            'mr_latent_shape': mr_data.shape,
            'mr_latent_path': mr_path,
            'original_mr_path': original_mr_path,
            'original_mr_exists': original_mr_exists,
            'channels': mr_data.shape[-1] if len(mr_data.shape) == 4 else 'Unknown',
            'test_mode': True
        }
        
        return info


def create_4channel_test_dataloader(opt):
    """
    Create DataLoader for 4-channel test dataset
    
    Args:
        opt: Options containing:
            - dataroot: Dataset root directory
            - batchSize: Batch size
            - nThreads: Number of data loading threads
            - serial_batches: Whether to use serial batching
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = FourChannelTestDataset(opt)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,  # No shuffling in test mode
        num_workers=int(opt.nThreads),
        drop_last=False,  # Don't drop last batch in test mode
        pin_memory=True
    )
    
    return dataloader


class FourChannelTestDataLoaderWrapper:
    """
    Wrapper class to match the expected interface from data_loader.py
    """
    
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = create_4channel_test_dataloader(opt)
        self.dataset = self.dataloader.dataset
    
    def load_data(self):
        """Return the dataloader"""
        return self.dataloader
    
    def __len__(self):
        """Return dataset size"""
        return len(self.dataset)
    
    def name(self):
        """Return dataset name"""
        return "FourChannelTestDataset"


def test_4channel_test_dataset():
    """Test function for 4-channel test dataset"""
    print("🧪 Testing 4-Channel Test Dataset...")
    
    # Mock options
    class MockOpt:
        def __init__(self):
            self.dataroot = "../datasets/test_channels"  # Your test directory
            self.phase = "test"
            self.input_nc = 4
            self.output_nc = 1
            self.batchSize = 1
            self.nThreads = 0
            self.serial_batches = True
            self.normalize_input = False
    
    opt = MockOpt()
    
    # Test dataset
    try:
        dataset = FourChannelTestDataset(opt)
        print(f"✅ Test dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   MR latent shape: {sample['A'].shape}")
        print(f"   CT shape: {sample['B'].shape}")
        print(f"   Sample name: {sample['sample_name']}")
        print(f"   Target shape: {sample['target_shape']}")
        print(f"   MR range: [{sample['A'].min():.3f}, {sample['A'].max():.3f}]")
        
        # Test sample info
        info = dataset.get_sample_info(0)
        print(f"✅ Sample info: {info}")
        
        # Test dataloader
        dataloader = create_4channel_test_dataloader(opt)
        batch = next(iter(dataloader))
        print(f"✅ Batch loaded:")
        print(f"   Batch MR shape: {batch['A'].shape}")
        print(f"   Batch CT shape: {batch['B'].shape}")
        print(f"   Sample names: {batch['sample_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_4channel_test_dataset()
