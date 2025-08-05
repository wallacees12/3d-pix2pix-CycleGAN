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
        
        # Add ONLY new 4-channel specific options (don't override existing ones)
        self.parser.add_argument('--lambda_L1', type=float, default=100.0, 
                               help='weight for L1 loss (important for medical imaging)')
    
    def parse(self):
        """Parse options and apply 4-channel specific logic"""
        opt = super().parse()
        
        # Set intelligent defaults only if user didn't specify them
        import sys
        cmd_args = ' '.join(sys.argv[1:])
        
        # Set dataset mode for 4-channel data (if not already set)
        if '--dataset_mode' not in cmd_args:
            opt.dataset_mode = 'four_channel'
        
        # Convert standard architectures to 4-channel equivalents (if user used standard names)
        if '--which_model_netG' not in cmd_args:
            # User didn't specify, use 4-channel default
            opt.which_model_netG = 'unet_4channel_128'
        else:
            # User specified - convert standard names to 4-channel equivalents
            arch_conversion = {
                'unet_128': 'unet_4channel_128',
                'unet_256': 'unet_4channel_256',
                'resnet_9blocks': 'resnet_4channel_9blocks',
                'resnet_6blocks': 'resnet_4channel_6blocks'
            }
            if opt.which_model_netG in arch_conversion:
                original_arch = opt.which_model_netG
                opt.which_model_netG = arch_conversion[original_arch]
                print(f"🔄 Converted architecture: {original_arch} → {opt.which_model_netG}")
        
        # Set sensible channel defaults only if user didn't specify them
        if '--input_nc' not in cmd_args:
            opt.input_nc = 4  # Default for 4-channel MR latent
            
        if '--output_nc' not in cmd_args:
            opt.output_nc = 1  # Default for synthetic CT (user can override with --output_nc 4 for latent-to-latent)
        
        # Print final configuration
        print(f"\n🔧 4-Channel Configuration:")
        print(f"   Input channels: {opt.input_nc}")
        print(f"   Output channels: {opt.output_nc}")
        print(f"   Architecture: {opt.which_model_netG}")
        print(f"   Dataset mode: {opt.dataset_mode}")
        print(f"   L1 weight: {opt.lambda_L1}")
        
        return opt
