import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
import pickle
import numpy as np


class NoduleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.pkl_file = os.path.join(opt.dataroot, "crops.pkl")
        # Updated for our data structure - no subdirectories needed
        self.scans_dir = opt.dataroot  # NPZ files are directly in dataroot
        self.heatmaps_dir = opt.dataroot  # NPZ files are directly in dataroot
        
        self.samples = pickle.load(open(self.pkl_file, 'rb'))
        
        # Our data structure is already flat, no need for nested list comprehension
        # self.samples = [ j for i in self.samples for j in i]

        random.shuffle(self.samples)
        
        self.scans = {}
        self.heatmaps = {}
        
        # Track if we've already printed dataset info (to avoid spam)
        self._info_printed = False


    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]

        sample = self.samples[index]
        
        # Our data structure uses 'name' instead of 'suid'
        sample_id = sample['name']
        
        #load scan and heatmap if it hasnt already been loaded
        if sample_id not in self.scans:
            # Load MR data (input - A)
            mr_path = os.path.join(self.scans_dir, sample['mr_path'])
            self.scans[sample_id] = np.load(mr_path)['data']
            
            # Load CT data (target - B)  
            ct_path = os.path.join(self.heatmaps_dir, sample['ct_path'])
            self.heatmaps[sample_id] = np.load(ct_path)['data']
            
        scan = self.scans[sample_id]  # MR data
        heatmap = self.heatmaps[sample_id]  # CT data

        #crop - our data structure has bounds as [z_start, z_end, y_start, y_end, x_start, x_end]
        b = sample['bounds']
        
        # For 3D UNet, we need patches that can be downsampled multiple times
        # The network uses 4x4x4 kernels with stride 2, so we need dimensions divisible by powers of 2
        # Let's use 64x64x64 patches which work well with the UNet architecture
        
        # Get original bounds
        z_start, z_end, y_start, y_end, x_start, x_end = b
        
        # Calculate available sizes
        z_size = z_end - z_start
        y_size = y_end - y_start  
        x_size = x_end - x_start
        
        # Use patches suitable for 3D UNet architecture
        # Option 1: Use full spatial resolution (32, 128, 128) with unet_128
        # Option 2: Pad to (32, 256, 256) for unet_256
        # Option 3: Use (32, 64, 64) patches for faster training
        
        # For your case with (32, 128, 128) data:
        # - Full data works perfectly with unet_128 
        # - Can pad to 256x256 for unet_256 if desired
        
        # Choose patch strategy based on available data
        if y_size >= 256 and x_size >= 256:
            # Use 256x256 patches for unet_256
            patch_z = min(32, z_size)  # Keep depth as is
            patch_y = 256
            patch_x = 256
            if not self._info_printed:
                print(f"Using 256x256 patches for unet_256 compatibility")
        elif y_size >= 128 and x_size >= 128:
            # Use full 128x128 for unet_128
            patch_z = min(32, z_size)
            patch_y = min(128, y_size)
            patch_x = min(128, x_size) 
            if not self._info_printed:
                print(f"Using {patch_y}x{patch_x} patches for unet_128 compatibility")
        else:
            # Fallback to 64x64 patches
            patch_z = min(32, z_size)
            patch_y = min(64, y_size)
            patch_x = min(64, x_size)
            if not self._info_printed:
                print(f"Using {patch_y}x{patch_x} smaller patches")
        
        # Center the patch within available bounds
        z_center = (z_start + z_end) // 2
        y_center = (y_start + y_end) // 2
        x_center = (x_start + x_end) // 2
        
        # Calculate patch bounds - handle cases where patch is larger than data
        patch_z_start = max(z_start, z_center - patch_z // 2)
        patch_z_end = min(z_end, patch_z_start + patch_z)
        
        # For Y dimension - handle padding if needed
        if patch_y > y_size:
            # Need padding - use all available data and pad later
            patch_y_start = y_start
            patch_y_end = y_end
            y_padding_needed = patch_y - y_size
        else:
            # No padding needed
            patch_y_start = max(y_start, y_center - patch_y // 2)
            patch_y_end = min(y_end, patch_y_start + patch_y)
            y_padding_needed = 0
            
        # For X dimension - handle padding if needed  
        if patch_x > x_size:
            # Need padding - use all available data and pad later
            patch_x_start = x_start
            patch_x_end = x_end
            x_padding_needed = patch_x - x_size
        else:
            # No padding needed
            patch_x_start = max(x_start, x_center - patch_x // 2)
            patch_x_end = min(x_end, patch_x_start + patch_x)
            x_padding_needed = 0
        
        scan_crop = scan[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        heatmap_crop = heatmap[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        # Apply padding if needed to reach target patch size
        if y_padding_needed > 0 or x_padding_needed > 0:
            pad_y_before = y_padding_needed // 2
            pad_y_after = y_padding_needed - pad_y_before
            pad_x_before = x_padding_needed // 2  
            pad_x_after = x_padding_needed - pad_x_before
            
            # Pad with zeros (background)
            pad_width = ((0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
            if len(scan_crop.shape) == 4:  # Has channel dimension
                pad_width = pad_width + ((0, 0),)
                
            scan_crop = np.pad(scan_crop, pad_width, mode='constant', constant_values=0)
            heatmap_crop = np.pad(heatmap_crop, pad_width[:3], mode='constant', constant_values=0)
            if not self._info_printed:
                print(f"Applied padding: y={y_padding_needed}, x={x_padding_needed}")

        # Handle different channel dimensions
        # Our MR data is (D, H, W, C) but we need (C, D, H, W) for PyTorch
        if len(scan_crop.shape) == 4:  # (D, H, W, C)
            scan_crop = scan_crop.transpose(3, 0, 1, 2)  # (C, D, H, W)
            if not self._info_printed:
                print(f"MR data: {scan_crop.shape} - Using all {scan_crop.shape[0]} channels")
        else:  # (D, H, W) - add channel dimension
            scan_crop = scan_crop[None, :]  # (1, D, H, W)
            if not self._info_printed:
                print(f"MR data: {scan_crop.shape} - Single channel")
            
        # CT data should be (D, H, W) - add channel dimension
        if len(heatmap_crop.shape) == 3:  # (D, H, W)
            heatmap_crop = heatmap_crop[None, :]  # (1, D, H, W)
            if not self._info_printed:
                print(f"CT data: {heatmap_crop.shape} - Single channel output")
        
        # Validate channel count (only print once)
        if not self._info_printed:
            input_channels = scan_crop.shape[0]
            if input_channels == 4:
                print(f"✅ Using full 4-channel latent representation")
            elif input_channels == 1:
                print(f"⚠️ Using single channel - consider using 4-channel data for better results")
            else:
                print(f"🔧 Using {input_channels} channels")
            
            # Mark that we've printed the info
            self._info_printed = True

        #convert to torch tensors with dimension [channel, z, x, y]
        scan_crop = torch.from_numpy(scan_crop.astype(np.float32))
        heatmap_crop = torch.from_numpy(heatmap_crop.astype(np.float32))
        
        return {
                'A' : scan_crop,   # MR input
                'B' : heatmap_crop # CT target
                }

    def __len__(self):
        return len(self.samples)

    def name(self):
        return 'NodulesDataset'

if __name__ == '__main__':
    #test
    n = NoduleDataset()
    n.initialize("datasets/nodules")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
