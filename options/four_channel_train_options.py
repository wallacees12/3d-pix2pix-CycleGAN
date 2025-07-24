#!/usr/bin/env python3
"""
4-Channel Training Options
Extended training options specifically for 4-channel input networks
"""

from options.train_options import TrainOptions

class FourChannelTrainOptions(TrainOptions):
    """Training options for 4-channel networks"""
    
    def initialize(self):
        # Initialize base options
        super().initialize()
        
        # 4-Channel specific options
        self.parser.add_argument('--input_nc', type=int, default=4, 
                               help='# of input image channels (4 for latent MR)')
        self.parser.add_argument('--output_nc', type=int, default=1, 
                               help='# of output image channels (1 for synthetic CT)')
        
        # Network architecture options for 4-channel
        self.parser.add_argument('--which_model_netG', type=str, default='unet_4channel_128',
                               choices=['unet_4channel_128', 'unet_4channel_256', 
                                      'resnet_4channel_6blocks', 'resnet_4channel_9blocks',
                                      'dense_4channel'],
                               help='selects model to use for netG')
        
        # Enhanced training options
        self.parser.add_argument('--lambda_L1', type=float, default=100.0, 
                               help='weight for L1 loss (important for medical imaging)')
        self.parser.add_argument('--lambda_perceptual', type=float, default=0.0,
                               help='weight for perceptual loss (experimental)')
        
        # Data augmentation
        self.parser.add_argument('--augment_data', action='store_true',
                               help='enable data augmentation (flips, rotations)')
        self.parser.add_argument('--normalize_input', action='store_true',
                               help='normalize input data to [-1,1] range')
        
        # Medical imaging specific options
        self.parser.add_argument('--hounsfield_range', type=str, default='-1024,3000',
                               help='Hounsfield unit range for CT normalization (min,max)')
        self.parser.add_argument('--clip_values', action='store_true',
                               help='clip output values to valid CT range')
        
        # Training stability
        self.parser.add_argument('--spectral_norm', action='store_true',
                               help='use spectral normalization in discriminator')
        self.parser.add_argument('--gradient_penalty', type=float, default=0.0,
                               help='weight for gradient penalty loss')
        
        # Monitoring and visualization
        self.parser.add_argument('--save_sample_freq', type=int, default=1000,
                               help='frequency of saving sample images during training')
        self.parser.add_argument('--validate_freq', type=int, default=5,
                               help='frequency of validation (epochs)')
        
        # Memory optimization
        self.parser.add_argument('--mixed_precision', action='store_true',
                               help='use mixed precision training (saves memory)')
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                               help='accumulate gradients over multiple steps')
        
        # Custom dataset options
        self.parser.add_argument('--use_4channel_dataset', action='store_true',
                               help='use custom 4-channel dataset loader')
        self.parser.add_argument('--dataset_mode', type=str, default='four_channel',
                               help='dataset mode for 4-channel data')
    
    def parse(self):
        """Parse options and set additional 4-channel specific settings"""
        opt = super().parse()
        
        # Force 4-channel settings
        opt.input_nc = 4
        opt.output_nc = 1
        
        # Parse Hounsfield range
        if hasattr(opt, 'hounsfield_range'):
            try:
                hu_min, hu_max = map(float, opt.hounsfield_range.split(','))
                opt.hu_min = hu_min
                opt.hu_max = hu_max
            except:
                print("⚠️ Warning: Invalid Hounsfield range format, using defaults")
                opt.hu_min = -1024
                opt.hu_max = 3000
        
        # Set dataset mode for 4-channel data
        if opt.use_4channel_dataset:
            opt.dataset_mode = 'four_channel'
        
        # Validate architecture choice
        valid_4ch_archs = ['unet_4channel_128', 'unet_4channel_256', 
                          'resnet_4channel_6blocks', 'resnet_4channel_9blocks',
                          'dense_4channel']
        if opt.which_model_netG not in valid_4ch_archs:
            print(f"⚠️ Warning: {opt.which_model_netG} is not a 4-channel architecture")
            print(f"   Available 4-channel architectures: {valid_4ch_archs}")
            print(f"   Defaulting to unet_4channel_128")
            opt.which_model_netG = 'unet_4channel_128'
        
        # Print 4-channel specific configuration
        print("\n🔧 4-Channel Training Configuration:")
        print(f"   Input channels: {opt.input_nc}")
        print(f"   Output channels: {opt.output_nc}")
        print(f"   Generator architecture: {opt.which_model_netG}")
        print(f"   L1 loss weight: {opt.lambda_L1}")
        print(f"   Data augmentation: {opt.augment_data}")
        print(f"   Input normalization: {opt.normalize_input}")
        print(f"   Hounsfield range: [{opt.hu_min}, {opt.hu_max}]")
        print(f"   Custom dataset: {opt.use_4channel_dataset}")
        
        return opt
