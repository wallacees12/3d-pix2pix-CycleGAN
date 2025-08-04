"""
3D Neural Networks for 4-Channel Input to Single-Channel Output
Designed for 4-channel latent representation to 1-channel synthetic CT generation

This module provides enhanced 3D architectures specifically designed to handle
4-channel input latent representations and generate single-channel synthetic CT volumes.
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    """Initialize network weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    """Get normalization layer for 3D operations"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, num_groups=8, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G_4channel(input_nc=4, output_nc=1, ngf=64, which_model_netG='unet_4channel_128', 
                     norm='batch', use_dropout=False, gpu_ids=[]):
    """
    Define Generator for 4-channel input to 1-channel output
    
    Args:
        input_nc (int): Number of input channels (4 for latent representation)
        output_nc (int): Number of output channels (1 for synthetic CT)
        ngf (int): Number of generator filters in first conv layer
        which_model_netG (str): Generator architecture type
        norm (str): Normalization type ('batch', 'instance', 'group')
        use_dropout (bool): Use dropout layers
        gpu_ids (list): GPU device IDs
    
    Returns:
        Generator network
    """
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'unet_4channel_128':
        netG = UnetGenerator4Channel(input_nc, output_nc, 5, ngf, 
                                   norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_4channel_256':
        netG = UnetGenerator4Channel(input_nc, output_nc, 6, ngf, 
                                   norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_4channel_9blocks':
        netG = ResnetGenerator4Channel(input_nc, output_nc, ngf, 
                                     norm_layer=norm_layer, use_dropout=use_dropout, 
                                     n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_4channel_6blocks':
        netG = ResnetGenerator4Channel(input_nc, output_nc, ngf, 
                                     norm_layer=norm_layer, use_dropout=use_dropout, 
                                     n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'dense_4channel':
        netG = DenseGenerator4Channel(input_nc, output_nc, ngf, 
                                    norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D_4channel(input_nc=1, ndf=64, which_model_netD='basic',
                     n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    """
    Define Discriminator for single-channel synthetic CT
    
    Args:
        input_nc (int): Number of input channels (1 for synthetic CT)
        ndf (int): Number of discriminator filters in first conv layer
        which_model_netD (str): Discriminator architecture type
        n_layers_D (int): Number of layers in discriminator
        norm (str): Normalization type
        use_sigmoid (bool): Use sigmoid activation in output
        gpu_ids (list): GPU device IDs
    
    Returns:
        Discriminator network
    """
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, 
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator3D(input_nc, ndf, n_layers_D, 
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'spectral':
        netD = SpectralDiscriminator3D(input_nc, ndf, n_layers_D, 
                                     norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    """Print network architecture and parameter count"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Loss Functions
##############################################################################

class GANLoss(nn.Module):
    """GAN Loss with LSGAN or regular GAN support"""
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                           (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                           (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


##############################################################################
# Generator Networks
##############################################################################

class UnetGenerator4Channel(nn.Module):
    """
    3D UNet Generator for 4-channel input to 1-channel output
    Optimized for latent representation to synthetic CT generation
    """
    def __init__(self, input_nc=4, output_nc=1, num_downs=5, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator4Channel, self).__init__()
        self.gpu_ids = gpu_ids
        
        # Input channel adaptation layer
        self.input_adapter = nn.Sequential(
            nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(False)  # Changed from inplace=True to avoid gradient issues
        )
        
        # Build UNet structure
        unet_block = UnetSkipConnectionBlock4Channel(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock4Channel(ngf * 8, ngf * 8, unet_block, 
                                                       norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock4Channel(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock4Channel(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock4Channel(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        
        # Output layer adapted for single channel (maintain input dimensions)
        self.unet_core = unet_block
        self.output_layer = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(False),  # Changed from inplace=True to avoid gradient issues
            nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1),
            nn.Tanh()  # For normalized CT values [-1, 1]
        )

    def forward(self, input):
        # Adapt 4-channel input
        adapted_input = self.input_adapter(input)
        
        # Pass through UNet
        unet_output = self.unet_core(adapted_input)
        
        # Generate single-channel output
        output = self.output_layer(unet_output)
        return output


class UnetSkipConnectionBlock4Channel(nn.Module):
    """
    UNet skip connection block optimized for 4-channel processing
    """
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock4Channel, self).__init__()
        self.outermost = outermost
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        if innermost:
            downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, False)  # Changed from inplace=True
            uprelu = nn.ReLU(False)  # Changed from inplace=True
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upnorm = norm_layer(outer_nc)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        elif outermost:
            downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, False)  # Changed from inplace=True
            uprelu = nn.ReLU(False)  # Changed from inplace=True
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, submodule, uprelu, upconv]
        else:
            downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            downnorm = norm_layer(inner_nc)
            downrelu = nn.LeakyReLU(0.2, False)  # Changed from inplace=True
            uprelu = nn.ReLU(False)  # Changed from inplace=True
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upnorm = norm_layer(outer_nc)
            
            if use_dropout:
                model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm, nn.Dropout(0.5)]
            else:
                model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class ResnetGenerator4Channel(nn.Module):
    """
    ResNet-based Generator for 4-channel input to 1-channel output
    """
    def __init__(self, input_nc=4, output_nc=1, ngf=64, norm_layer=nn.BatchNorm3d, 
                 use_dropout=False, n_blocks=6, gpu_ids=[]):
        super(ResnetGenerator4Channel, self).__init__()
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # Input layer with 4-channel adaptation
        model = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # ResNet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock4Channel(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, 
                                 padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        # Output layer for single channel
        model += [
            nn.ReflectionPad3d(3),
            nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetBlock4Channel(nn.Module):
    """ResNet block optimized for 4-channel processing"""
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock4Channel, self).__init__()
        
        conv_block = []
        conv_block += [nn.ReflectionPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim),
                      nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DenseGenerator4Channel(nn.Module):
    """
    Dense Generator with attention mechanism for 4-channel input
    """
    def __init__(self, input_nc=4, output_nc=1, ngf=64, norm_layer=nn.BatchNorm3d, 
                 use_dropout=False, gpu_ids=[]):
        super(DenseGenerator4Channel, self).__init__()
        self.gpu_ids = gpu_ids
        
        # Channel attention for 4-channel input
        self.channel_attention = ChannelAttention4Channel(input_nc)
        
        # Dense blocks
        self.dense_block1 = DenseBlock4Channel(input_nc, ngf, norm_layer)
        self.dense_block2 = DenseBlock4Channel(ngf, ngf * 2, norm_layer)
        self.dense_block3 = DenseBlock4Channel(ngf * 2, ngf * 4, norm_layer)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv3d(ngf * 4, ngf, kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, output_nc, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, input):
        # Apply channel attention
        attended_input = self.channel_attention(input)
        
        # Dense processing
        x1 = self.dense_block1(attended_input)
        x2 = self.dense_block2(x1)
        x3 = self.dense_block3(x2)
        
        # Generate output
        output = self.output_proj(x3)
        return output


class ChannelAttention4Channel(nn.Module):
    """Channel attention module for 4-channel input"""
    def __init__(self, in_channels):
        super(ChannelAttention4Channel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class DenseBlock4Channel(nn.Module):
    """Dense block for feature reuse"""
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DenseBlock4Channel, self).__init__()
        
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(True),
            nn.Conv3d(in_channels, out_channels // 2, 1)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels // 2),
            nn.ReLU(True),
            nn.Conv3d(out_channels // 2, out_channels // 2, 3, padding=1)
        )
        
        self.conv3 = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        concat = torch.cat([out1, out2], 1)
        out3 = self.conv3(concat)
        return out3


##############################################################################
# Discriminator Networks
##############################################################################

class NLayerDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator for single-channel CT"""
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, 
                 use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator3D, self).__init__()
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, 
                         padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, 
                     padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class SpectralDiscriminator3D(nn.Module):
    """3D Discriminator with spectral normalization for training stability"""
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, 
                 use_sigmoid=False, gpu_ids=[]):
        super(SpectralDiscriminator3D, self).__init__()
        self.gpu_ids = gpu_ids
        
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        
        sequence = [
            nn.utils.spectral_norm(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                                                kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                                            kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.utils.spectral_norm(nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


##############################################################################
# Utility Functions
##############################################################################

def test_networks_4channel():
    """Test function to verify 4-channel networks work correctly"""
    print("Testing 4-Channel 3D Networks...")
    
    # Test input: batch_size=1, channels=4, depth=32, height=128, width=128
    test_input = torch.randn(1, 4, 32, 128, 128)
    
    # Test UNet Generator
    print("\n1. Testing UNet Generator 4-Channel...")
    unet_gen = UnetGenerator4Channel(input_nc=4, output_nc=1, num_downs=5, ngf=64)
    unet_output = unet_gen(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {unet_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in unet_gen.parameters())}")
    
    # Test ResNet Generator
    print("\n2. Testing ResNet Generator 4-Channel...")
    resnet_gen = ResnetGenerator4Channel(input_nc=4, output_nc=1, ngf=64, n_blocks=6)
    resnet_output = resnet_gen(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {resnet_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in resnet_gen.parameters())}")
    
    # Test Dense Generator
    print("\n3. Testing Dense Generator 4-Channel...")
    dense_gen = DenseGenerator4Channel(input_nc=4, output_nc=1, ngf=64)
    dense_output = dense_gen(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {dense_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dense_gen.parameters())}")
    
    # Test Discriminator
    print("\n4. Testing Discriminator...")
    disc = NLayerDiscriminator3D(input_nc=1, ndf=64, n_layers=3)
    disc_output = disc(unet_output)
    print(f"   Input shape: {unet_output.shape}")
    print(f"   Output shape: {disc_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in disc.parameters())}")
    
    print("\n✅ All 4-channel networks tested successfully!")
    return True


if __name__ == "__main__":
    test_networks_4channel()
