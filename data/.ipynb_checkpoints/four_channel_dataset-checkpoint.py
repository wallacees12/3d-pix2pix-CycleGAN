"""
4-Channel Dataset Loader for 3D Medical Imaging
Specifically designed to load 4-channel MR latent → 1-channel CT datasets
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import random

class FourChannelDataset(Dataset):
    """
    Dataset class for 4-channel MR latent to 1-channel CT conversion
    
    Compatible with datasets created by 4Channel_data.ipynb
    """
    
    def __init__(self, opt):
        """
        Initialize 4-channel dataset
        
        Args:
            opt: Options containing dataset configuration
                - dataroot: Root directory containing the dataset
                - phase: 'train', 'val', or 'test'
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
        
        print(f"📊 Loaded {len(self.samples)} samples from {crops_file}")
        
        # Validate sample data
        for i, sample in enumerate(self.samples):
            required_keys = ['name', 'mr_path', 'ct_path', 'bounds']
            for key in required_keys:
                if key not in sample:
                    raise KeyError(f"Sample {i} missing required key: {key}")
        
        # Data augmentation settings
        self.use_augmentation = opt.phase == 'train' and not opt.no_flip
        
        print(f"✅ 4-Channel dataset initialized:")
        print(f"   Root: {self.root}")
        print(f"   Phase: {getattr(opt, 'phase', 'train')}")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Augmentation: {self.use_augmentation}")
    
    def name(self):
        """Return dataset name"""
        return 'FourChannelDataset'
    
    def initialize(self, opt):
        """Initialize method for compatibility with base dataset interface"""
        # Dataset is already initialized in __init__, this is for compatibility
        pass
    
    def __len__(self):
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a data sample
        
        Returns:
            dict containing:
                - A: 4-channel MR latent tensor [4, D, H, W]
                - B: 1-channel CT tensor [1, D, H, W]
                - A_paths: MR file path
                - B_paths: CT file path
        """
        sample = self.samples[index]
        
        # Load MR data (4-channel)
        mr_path = os.path.join(self.root, sample['mr_path'])
        if not os.path.exists(mr_path):
            raise FileNotFoundError(f"MR file not found: {mr_path}")
        
        mr_data = np.load(mr_path, allow_pickle=True)['data']  # Shape: [D, H, W, C] or [D, H, W]
        
        # Load CT data (1-channel) - only if CT path is provided
        ct_path = None
        if sample['ct_path'] is not None:
            ct_path = os.path.join(self.root, sample['ct_path'])
            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT file not found: {ct_path}")
            
            ct_data = np.load(ct_path, allow_pickle=True)['data']  # Shape: [D, H, W]
        else:
            # For test datasets without ground truth CT
            ct_data = None
        
        # Handle channel dimensions
        if len(mr_data.shape) == 4:
            # Already has channels [D, H, W, C] → [C, D, H, W]
            mr_data = np.transpose(mr_data, (3, 0, 1, 2))
            if mr_data.shape[0] != 4:
                raise ValueError(f"Expected 4 MR channels, got {mr_data.shape[0]}")
        elif len(mr_data.shape) == 3:
            # Add channel dimension [D, H, W] → [1, D, H, W]
            mr_data = mr_data[np.newaxis, ...]
            print(f"⚠️ Warning: Only single channel MR data available")
        
        # Ensure CT is single channel [D, H, W] → [1, D, H, W]
        if ct_data is not None:
            if len(ct_data.shape) == 3:
                ct_data = ct_data[np.newaxis, ...]
            elif len(ct_data.shape) == 4:
                # Take first channel if multiple channels exist
                ct_data = ct_data[0:1, ...]
        
        # Apply cropping if specified
        bounds = sample.get('bounds', None)
        if bounds and len(bounds) == 6:
            z_start, z_end, y_start, y_end, x_start, x_end = bounds
            mr_data = mr_data[:, z_start:z_end, y_start:y_end, x_start:x_end]
            if ct_data is not None:
                ct_data = ct_data[:, z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Data augmentation
        if self.use_augmentation and ct_data is not None:
            mr_data, ct_data = self._apply_augmentation(mr_data, ct_data)
        
        # Convert to tensors
        mr_tensor = torch.from_numpy(mr_data.astype(np.float32))
        
        if ct_data is not None:
            ct_tensor = torch.from_numpy(ct_data.astype(np.float32))
        else:
            # Create dummy tensor for test phase
            ct_tensor = torch.zeros(1, *mr_data.shape[1:], dtype=torch.float32)
        
        # Normalize data if needed
        if hasattr(self.opt, 'normalize_input') and self.opt.normalize_input:
            mr_tensor = self._normalize_tensor(mr_tensor)
            if ct_data is not None:
                ct_tensor = self._normalize_tensor(ct_tensor)
        
        return {
            'A': mr_tensor,           # 4-channel MR latent
            'B': ct_tensor,           # 1-channel CT (or dummy zeros)
            'A_paths': mr_path,       # MR file path
            'B_paths': sample['ct_path']  # CT file path (can be None)
        }
    
    def _apply_augmentation(self, mr_data, ct_data):
        """
        Apply data augmentation
        
        Args:
            mr_data: MR data [C, D, H, W]
            ct_data: CT data [1, D, H, W]
        
        Returns:
            Augmented mr_data, ct_data
        """
        # Random flipping
        if random.random() > 0.5:
            # Flip along width dimension
            mr_data = np.flip(mr_data, axis=3).copy()
            ct_data = np.flip(ct_data, axis=3).copy()
        
        if random.random() > 0.5:
            # Flip along height dimension
            mr_data = np.flip(mr_data, axis=2).copy()
            ct_data = np.flip(ct_data, axis=2).copy()
        
        # Random rotation (90-degree increments)
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            mr_data = np.rot90(mr_data, k=k, axes=(2, 3)).copy()
            ct_data = np.rot90(ct_data, k=k, axes=(2, 3)).copy()
        
        return mr_data, ct_data
    
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
        """Get information about a specific sample"""
        sample = self.samples[index]
        
        # Load data to get shapes
        mr_path = os.path.join(self.root, sample['mr_path'])
        ct_path = os.path.join(self.root, sample['ct_path'])
        
        mr_data = np.load(mr_path)['data']
        ct_data = np.load(ct_path)['data']
        
        info = {
            'name': sample['name'],
            'mr_shape': mr_data.shape,
            'ct_shape': ct_data.shape,
            'mr_path': mr_path,
            'ct_path': ct_path,
            'bounds': sample.get('bounds', 'Full volume'),
            'channels': sample.get('channel_count', 'Unknown')
        }
        
        return info


def create_4channel_dataloader(opt):
    """
    Create DataLoader for 4-channel dataset
    
    Args:
        opt: Options containing:
            - dataroot: Dataset root directory
            - batchSize: Batch size
            - nThreads: Number of data loading threads
            - serial_batches: Whether to use serial batching
            - phase: Dataset phase ('train', 'val', 'test')
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = FourChannelDataset(opt)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches and opt.phase == 'train',
        num_workers=int(opt.nThreads),
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader


class FourChannelDataLoaderWrapper:
    """
    Wrapper class to match the expected interface from data_loader.py
    """
    
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = create_4channel_dataloader(opt)
        self.dataset = self.dataloader.dataset
    
    def load_data(self):
        """Return the dataloader"""
        return self.dataloader
    
    def __len__(self):
        """Return dataset size"""
        return len(self.dataset)
    
    def name(self):
        """Return dataset name"""
        return "FourChannelDataset"


def test_4channel_dataset():
    """Test function for 4-channel dataset"""
    print("🧪 Testing 4-Channel Dataset...")
    
    # Mock options
    class MockOpt:
        def __init__(self):
            self.dataroot = "../datasets/all_channels"
            self.phase = "train"
            self.input_nc = 4
            self.output_nc = 1
            self.batchSize = 1
            self.nThreads = 0
            self.serial_batches = True
            self.no_flip = False
            self.normalize_input = False
    
    opt = MockOpt()
    
    # Test dataset
    try:
        dataset = FourChannelDataset(opt)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   MR shape: {sample['A'].shape}")
        print(f"   CT shape: {sample['B'].shape}")
        print(f"   MR range: [{sample['A'].min():.3f}, {sample['A'].max():.3f}]")
        print(f"   CT range: [{sample['B'].min():.3f}, {sample['B'].max():.3f}]")
        
        # Test sample info
        info = dataset.get_sample_info(0)
        print(f"✅ Sample info: {info}")
        
        # Test dataloader
        dataloader = create_4channel_dataloader(opt)
        batch = next(iter(dataloader))
        print(f"✅ Batch loaded:")
        print(f"   Batch MR shape: {batch['A'].shape}")
        print(f"   Batch CT shape: {batch['B'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        return False


if __name__ == "__main__":
    test_4channel_dataset()
