#!/usr/bin/env python3
"""
4-Channel Training Options
Extended training options specifically for 4-channel input networks
"""

from options.train_options import TrainOptions

class FourChannelTrainOptions(TrainOptions):
    """Training options for 4-channel networks"""
    
    def initialize(self):
        # Initialize base options first
        super().initialize()
        
        # Add only NEW 4-channel specific options (don't duplicate existing ones)
        self.parser.add_argument('--lambda_L1', type=float, default=100.0, 
                               help='weight for L1 loss (important for medical imaging)')
        
        # Custom dataset options
        self.parser.add_argument('--use_4channel_dataset', action='store_true',
                               help='use custom 4-channel dataset loader')
    
    def parse(self):
        """Parse options and set 4-channel specific defaults"""
        opt = super().parse()
        
        # Set 4-channel defaults for existing arguments
        if not hasattr(opt, '_input_nc_set_by_user'):
            opt.input_nc = 4  # Default to 4 channels for MR latent
            
        if not hasattr(opt, '_output_nc_set_by_user'):
            opt.output_nc = 1  # Default to 1 channel for synthetic CT
            
        # Set sensible defaults for 4-channel architectures
        if opt.which_model_netG in ['resnet_9blocks', 'resnet_6blocks', 'unet_128', 'unet_256']:
            # Convert to 4-channel equivalent
            arch_mapping = {
                'resnet_9blocks': 'resnet_4channel_9blocks',
                'resnet_6blocks': 'resnet_4channel_6blocks', 
                'unet_128': 'unet_4channel_128',
                'unet_256': 'unet_4channel_256'
            }
            opt.which_model_netG = arch_mapping.get(opt.which_model_netG, 'unet_4channel_256')
            print(f"🔄 Auto-selected 4-channel architecture: {opt.which_model_netG}")
        
        # Set dataset mode for 4-channel data
        if opt.use_4channel_dataset or opt.input_nc == 4:
            opt.dataset_mode = 'four_channel'
            
        # Print configuration
        print(f"\n🔧 4-Channel Configuration:")
        print(f"   Input channels: {opt.input_nc}")
        print(f"   Output channels: {opt.output_nc}")
        print(f"   Architecture: {opt.which_model_netG}")
        print(f"   Dataset mode: {opt.dataset_mode}")
        print(f"   L1 weight: {opt.lambda_L1}")
        
        return opt
