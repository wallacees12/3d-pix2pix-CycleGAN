#!/usr/bin/env python3
"""
4-Channel 3D Pix2Pix Training Script
Trains a multi-channel input to multi-channel output model for latent translation

This script supports:
- MR latent → synthetic CT generation (4→1 channels)  
- Latent-to-latent translation (4→4 channels)
- Any custom channel configuration
"""

import time
import os
import torch
from options.four_channel_train_options import FourChannelTrainOptions
from data.data_loader import CreateDataLoader
import numpy as np
from tqdm import tqdm

def create_4channel_model(opt):
    """Create multi-channel model using our custom networks"""
    # Import our multi-channel networks
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel, define_D_4channel, GANLoss
        print("✅ Successfully imported multi-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import multi-channel networks: {e}")
        print("   Make sure networks_3d_4channel.py is in the models/ directory")
        raise
    
    # Create multi-channel generator (configurable input_nc → output_nc)
    netG = define_G_4channel(
        input_nc=opt.input_nc,               # Configurable input channels (default: 4)
        output_nc=opt.output_nc,             # Configurable output channels (1 for CT, 4 for latent-to-latent)
        ngf=opt.ngf,                         # Number of generator filters
        which_model_netG=opt.which_model_netG, # Architecture type
        norm=opt.norm,                       # Normalization type
        use_dropout=not opt.no_dropout,      # Dropout usage
        gpu_ids=opt.gpu_ids                  # GPU devices
    )
    
    # Create discriminator for output channels
    netD = define_D_4channel(
        input_nc=opt.output_nc,              # Discriminator input = generator output channels
        ndf=opt.ndf,                         # Number of discriminator filters
        which_model_netD=opt.which_model_netD, # Discriminator architecture
        n_layers_D=opt.n_layers_D,          # Number of discriminator layers
        norm=opt.norm,                       # Normalization type
        use_sigmoid=opt.no_lsgan,            # Sigmoid activation
        gpu_ids=opt.gpu_ids                  # GPU devices
    )
    
    print(f"🏗️ Created networks:")
    print(f"   Generator: {opt.input_nc} → {opt.output_nc} channels")
    print(f"   Discriminator: {opt.output_nc} channels input")
    
    # Create loss function
    criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor)
    criterionL1 = torch.nn.L1Loss()
    
    # Create custom model class
    class Pix2Pix4ChannelModel:
        def __init__(self):
            self.netG = netG
            self.netD = netD
            self.criterionGAN = criterionGAN
            self.criterionL1 = criterionL1
            self.lambda_L1 = opt.lambda_L1
            self.gpu_ids = opt.gpu_ids
            
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            # Learning rate schedulers
            self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            )
            self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D, lr_lambda=lambda epoch: 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            )
            
            # Loss storage
            self.loss_G = 0
            self.loss_D = 0
            self.loss_G_GAN = 0
            self.loss_G_L1 = 0
            self.loss_D_real = 0
            self.loss_D_fake = 0
        
        def set_input(self, input):
            """Set input data for the model"""
            if 'A' in input and 'B' in input:
                # Standard paired data
                self.real_A = input['A']  # 4-channel MR latent
                self.real_B = input['B']  # 1-channel CT
            elif 'data' in input:
                # Custom 4-channel dataset format
                data = input['data']
                if data.shape[1] == 4:  # Check if we have 4 channels
                    self.real_A = data      # 4-channel MR latent
                    # For unpaired training, we'll need to handle this differently
                    self.real_B = None
                else:
                    print(f"⚠️ Warning: Expected 4 channels, got {data.shape[1]}")
                    self.real_A = data
                    self.real_B = None
            
            # Move to GPU if available
            if self.gpu_ids:
                if self.real_A is not None:
                    self.real_A = self.real_A.cuda(self.gpu_ids[0])
                if self.real_B is not None:
                    self.real_B = self.real_B.cuda(self.gpu_ids[0])
        
        def forward(self):
            """Forward pass through generator"""
            self.fake_B = self.netG(self.real_A)
        
        def backward_D(self):
            """Backward pass for discriminator"""
            if self.real_B is None:
                # For unpaired training, skip discriminator updates
                self.loss_D = 0
                return
            
            # Real
            pred_real = self.netD(self.real_B)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            
            # Fake
            pred_fake = self.netD(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # Combined loss
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
            self.loss_D.backward()
        
        def backward_G(self):
            """Backward pass for generator"""
            # GAN loss
            if self.real_B is not None:
                pred_fake = self.netD(self.fake_B)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                
                # L1 loss
                self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
                
                # Combined generator loss
                self.loss_G = self.loss_G_GAN + self.loss_G_L1
            else:
                # For unpaired training, use only reconstruction loss or other objectives
                # You might want to implement cycle consistency or other losses here
                self.loss_G_GAN = 0
                self.loss_G_L1 = 0
                self.loss_G = torch.tensor(0.0, requires_grad=True).cuda(self.gpu_ids[0]) if self.gpu_ids else torch.tensor(0.0, requires_grad=True)
            
            self.loss_G.backward()
        
        def optimize_parameters(self):
            """Optimize both generator and discriminator"""
            self.forward()
            
            # Update D
            if self.real_B is not None:
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()
            
            # Update G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        
        def get_current_errors(self):
            """Get current training errors"""
            return {
                'G_GAN': float(self.loss_G_GAN) if hasattr(self.loss_G_GAN, 'item') else 0,
                'G_L1': float(self.loss_G_L1) if hasattr(self.loss_G_L1, 'item') else 0,
                'D_real': float(self.loss_D_real) if hasattr(self.loss_D_real, 'item') else 0,
                'D_fake': float(self.loss_D_fake) if hasattr(self.loss_D_fake, 'item') else 0
            }
        
        def get_current_visuals(self):
            """Get current visual results"""
            visuals = {}
            if hasattr(self, 'real_A') and self.real_A is not None:
                # Show first channel of 4-channel input
                visuals['real_A'] = self.real_A[:, 0:1, :, :, :]  # Take first channel only for visualization
            if hasattr(self, 'fake_B') and self.fake_B is not None:
                visuals['fake_B'] = self.fake_B
            if hasattr(self, 'real_B') and self.real_B is not None:
                visuals['real_B'] = self.real_B
            return visuals
        
        def save(self, label):
            """Save model"""
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.netG.state_dict(), os.path.join(save_dir, f'netG_{label}.pth'))
            if self.real_B is not None:  # Only save discriminator if we have paired data
                torch.save(self.netD.state_dict(), os.path.join(save_dir, f'netD_{label}.pth'))
        
        def update_learning_rate(self):
            """Update learning rates"""
            self.scheduler_G.step()
            if self.real_B is not None:
                self.scheduler_D.step()
    
    return Pix2Pix4ChannelModel()

def main():
    # Parse training options using 4-channel specific options
    opt = FourChannelTrainOptions().parse()
    
    # Create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'📊 Dataset size: {dataset_size}')
    
    # Create model
    print("🏗️ Creating multi-channel model...")
    model = create_4channel_model(opt)
    
    # Training parameters
    total_steps = 0
    start_epoch = opt.epoch_count
    
    print(f"🚀 Starting multi-channel training...")
    print(f"   Epochs: {start_epoch} to {opt.niter + opt.niter_decay}")
    print(f"   Batch size: {opt.batchSize}")
    print(f"   Learning rate: {opt.lr}")
    print(f"   Input channels: {opt.input_nc}")
    print(f"   Output channels: {opt.output_nc}")
    print(f"   Architecture: {opt.which_model_netG}")
    print(f"   L1 weight: {opt.lambda_L1}")
    
    # Training loop
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # Progress bar for epoch
        pbar = tqdm(dataset, desc=f'Epoch {epoch}', unit='batch')
        
        for i, data in enumerate(pbar):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            # Set input and optimize
            model.set_input(data)
            model.optimize_parameters()
            
            # Update progress bar with current losses
            errors = model.get_current_errors()
            pbar.set_postfix({
                'G_GAN': f"{errors['G_GAN']:.3f}",
                'G_L1': f"{errors['G_L1']:.3f}",
                'D_real': f"{errors['D_real']:.3f}",
                'D_fake': f"{errors['D_fake']:.3f}"
            })
            
            # Print losses
            if total_steps % opt.print_freq == 0:
                tqdm.write(f'📊 Step {total_steps}, Epoch {epoch}[{epoch_iter}/{dataset_size}]: ' + 
                          ', '.join([f'{k}: {v:.4f}' for k, v in errors.items()]))
        
        # Save model
        if epoch % opt.save_epoch_freq == 0:
            print(f'💾 Saving model at epoch {epoch}')
            model.save('latest')
            model.save(f'epoch_{epoch}')
        
        # Update learning rates
        model.update_learning_rate()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'⏱️ Epoch {epoch} completed in {epoch_time:.2f}s')
    

if __name__ == '__main__':
    main()
