#!/usr/bin/env python3
"""
4-Channel 3D Pix2Pix Training Script - High Resolution
Trains a 4-channel input to 1-channel output model for MR latent → synthetic CT generation

Updated for high-resolution latent data: (4, 64, 256, 256) → (64, 256, 256)
"""

import time
import os
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import numpy as np
from tqdm import tqdm

def create_4channel_model(opt):
    """Create 4-channel model using our custom networks"""
    # Import our 4-channel networks
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    try:
        from models.networks_3d_4channel import define_G_4channel, define_D_4channel, GANLoss
        print("✅ Successfully imported 4-channel networks")
    except ImportError as e:
        print(f"❌ Failed to import 4-channel networks: {e}")
        print("   Make sure networks_3d_4channel.py is in the models/ directory")
        raise
    
    # Create 4-channel generator (4 input channels → 1 output channel)
    # Updated for high-resolution processing
    netG = define_G_4channel(
        input_nc=4,                           # 4-channel MR latent input
        output_nc=1,                          # Single-channel synthetic CT output
        ngf=opt.ngf,                         # Number of generator filters
        which_model_netG=opt.which_model_netG, # Architecture type
        norm=opt.norm,                       # Normalization type
        use_dropout=not opt.no_dropout,      # Dropout usage
        gpu_ids=opt.gpu_ids               # GPU devices
    )
    
    # Create discriminator for single-channel CT - adapted for high-res
    netD = define_D_4channel(
        input_nc=opt.output_nc,              # Single-channel CT input to discriminator
        ndf=opt.ndf,                         # Number of discriminator filters
        which_model_netD=opt.which_model_netD, # Discriminator architecture
        n_layers_D=opt.n_layers_D,          # Number of discriminator layers
        norm=opt.norm,                       # Normalization type
        use_sigmoid=opt.no_lsgan,            # Sigmoid activation
        gpu_ids=opt.gpu_ids                 # GPU devices
    )
    
    # Create loss function
    criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor)
    criterionL1 = torch.nn.L1Loss()
    
    # Add perceptual loss for high-resolution training (optional)
    if getattr(opt, 'use_perceptual_loss', False):
        from models.perceptual_loss import PerceptualLoss
        criterionPerceptual = PerceptualLoss(gpu_ids=opt.gpu_ids)
    else:
        criterionPerceptual = None
    
    # Create custom model class
    class Pix2Pix4ChannelModel:
        def __init__(self):
            self.netG = netG
            self.netD = netD
            self.criterionGAN = criterionGAN
            self.criterionL1 = criterionL1
            self.criterionPerceptual = criterionPerceptual
            self.lambda_L1 = opt.lambda_L1
            self.lambda_perceptual = getattr(opt, 'lambda_perceptual', 1.0)
            self.gpu_ids = opt.gpu_ids
            
            # Initialize optimizers with adjusted learning rates for high-res
            base_lr = opt.lr
            if hasattr(opt, 'lr_scale_highres'):
                base_lr *= opt.lr_scale_highres
            
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), 
                lr=base_lr, 
                betas=(opt.beta1, 0.999),
                weight_decay=getattr(opt, 'weight_decay', 1e-4)  # Add weight decay for stability
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), 
                lr=base_lr, 
                betas=(opt.beta1, 0.999),
                weight_decay=getattr(opt, 'weight_decay', 1e-4)
            )
            
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
            self.loss_G_perceptual = 0
            self.loss_D_real = 0
            self.loss_D_fake = 0
            
            # Memory optimization for high-res
            self.use_mixed_precision = getattr(opt, 'mixed_precision', True)
            if self.use_mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()
        
        def set_input(self, input):
            """Set input data for the model - enhanced for high-res validation"""
            if 'A' in input and 'B' in input:
                # Standard paired data
                self.real_A = input['A']  # 4-channel MR latent
                self.real_B = input['B']  # 1-channel CT
            elif 'data' in input:
                # Custom 4-channel dataset format
                data = input['data']
                if len(data.shape) == 5 and data.shape[1] == 4:  # (B, 4, D, H, W)
                    self.real_A = data      # 4-channel MR latent
                    # For unpaired training, we'll need to handle this differently
                    self.real_B = None
                elif len(data.shape) == 4 and data.shape[0] == 4:  # (4, D, H, W) - single sample
                    self.real_A = data.unsqueeze(0)  # Add batch dimension
                    self.real_B = None
                else:
                    print(f"⚠️ Warning: Expected shape (B,4,D,H,W) or (4,D,H,W), got {data.shape}")
                    self.real_A = data
                    self.real_B = None
            
            # Validate high-resolution dimensions
            if self.real_A is not None:
                expected_shape = (4, 64, 256, 256)  # Without batch dimension
                actual_shape = self.real_A.shape[1:] if len(self.real_A.shape) == 5 else self.real_A.shape
                
                if actual_shape != expected_shape:
                    print(f"⚠️ Shape mismatch! Expected {expected_shape}, got {actual_shape}")
                    # Handle shape mismatch - you might want to resize or crop
                    if actual_shape[0] == 4:  # Correct number of channels
                        print(f"   Continuing with shape {actual_shape}")
                    else:
                        raise ValueError(f"Incorrect number of channels: {actual_shape[0]}")
            
            # Move to GPU if available
            if self.gpu_ids:
                if self.real_A is not None:
                    self.real_A = self.real_A.cuda(self.gpu_ids[0])
                if self.real_B is not None:
                    self.real_B = self.real_B.cuda(self.gpu_ids[0])
        
        def forward(self):
            """Forward pass through generator - with memory optimization"""
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    self.fake_B = self.netG(self.real_A)
            else:
                self.fake_B = self.netG(self.real_A)
        
        def backward_D(self):
            """Backward pass for discriminator - optimized for high-res"""
            if self.real_B is None:
                # For unpaired training, skip discriminator updates
                self.loss_D = 0
                return
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Real
                    pred_real = self.netD(self.real_B)
                    self.loss_D_real = self.criterionGAN(pred_real, True)
                    
                    # Fake
                    pred_fake = self.netD(self.fake_B.detach())
                    self.loss_D_fake = self.criterionGAN(pred_fake, False)
                    
                    # Combined loss
                    self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
                
                self.scaler.scale(self.loss_D).backward()
            else:
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
            """Backward pass for generator - enhanced with perceptual loss"""
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # GAN loss
                    if self.real_B is not None:
                        pred_fake = self.netD(self.fake_B)
                        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                        
                        # L1 loss
                        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
                        
                        # Perceptual loss for high-resolution quality
                        if self.criterionPerceptual is not None:
                            self.loss_G_perceptual = self.criterionPerceptual(self.fake_B, self.real_B) * self.lambda_perceptual
                        else:
                            self.loss_G_perceptual = 0
                        
                        # Combined generator loss
                        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
                    else:
                        # For unpaired training
                        self.loss_G_GAN = 0
                        self.loss_G_L1 = 0
                        self.loss_G_perceptual = 0
                        self.loss_G = torch.tensor(0.0, requires_grad=True).cuda(self.gpu_ids[0]) if self.gpu_ids else torch.tensor(0.0, requires_grad=True)
                
                self.scaler.scale(self.loss_G).backward()
            else:
                # GAN loss
                if self.real_B is not None:
                    pred_fake = self.netD(self.fake_B)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                    
                    # L1 loss
                    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
                    
                    # Perceptual loss
                    if self.criterionPerceptual is not None:
                        self.loss_G_perceptual = self.criterionPerceptual(self.fake_B, self.real_B) * self.lambda_perceptual
                    else:
                        self.loss_G_perceptual = 0
                    
                    # Combined generator loss
                    self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
                else:
                    self.loss_G_GAN = 0
                    self.loss_G_L1 = 0
                    self.loss_G_perceptual = 0
                    self.loss_G = torch.tensor(0.0, requires_grad=True).cuda(self.gpu_ids[0]) if self.gpu_ids else torch.tensor(0.0, requires_grad=True)
                
                self.loss_G.backward()
        
        def optimize_parameters(self):
            """Optimize both generator and discriminator - with mixed precision"""
            self.forward()
            
            # Update D
            if self.real_B is not None:
                self.optimizer_D.zero_grad()
                self.backward_D()
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer_D)
                else:
                    self.optimizer_D.step()
            
            # Update G
            self.optimizer_G.zero_grad()
            self.backward_G()
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                self.optimizer_G.step()
        
        def get_current_errors(self):
            """Get current training errors - including perceptual loss"""
            return {
                'G_GAN': float(self.loss_G_GAN) if hasattr(self.loss_G_GAN, 'item') else 0,
                'G_L1': float(self.loss_G_L1) if hasattr(self.loss_G_L1, 'item') else 0,
                'G_perceptual': float(self.loss_G_perceptual) if hasattr(self.loss_G_perceptual, 'item') else 0,
                'D_real': float(self.loss_D_real) if hasattr(self.loss_D_real, 'item') else 0,
                'D_fake': float(self.loss_D_fake) if hasattr(self.loss_D_fake, 'item') else 0
            }
        
        def get_current_visuals(self):
            """Get current visual results - optimized for high-res display"""
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
    # Custom argument parsing for 4-channel specific options
    import argparse
    import sys
    
    # Remove custom 4-channel arguments from sys.argv to avoid conflicts with TrainOptions
    custom_args = {}
    filtered_argv = [sys.argv[0]]  # Keep script name
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--lambda_L1':
            custom_args['lambda_L1'] = float(sys.argv[i + 1])
            i += 2
        elif arg == '--lambda_perceptual':
            custom_args['lambda_perceptual'] = float(sys.argv[i + 1])
            i += 2
        elif arg == '--use_4channel_dataset':
            custom_args['use_4channel_dataset'] = True
            i += 1
        elif arg == '--mixed_precision':
            custom_args['mixed_precision'] = True
            i += 1
        elif arg == '--gradient_checkpointing':
            custom_args['gradient_checkpointing'] = True
            i += 1
        elif arg == '--use_perceptual_loss':
            custom_args['use_perceptual_loss'] = True
            i += 1
        else:
            filtered_argv.append(arg)
            i += 1
    
    # Temporarily replace sys.argv for TrainOptions parsing
    original_argv = sys.argv
    sys.argv = filtered_argv
    
    # Parse training options with 4-channel specific additions
    opt = TrainOptions().parse()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    # Add custom 4-channel options with high-res defaults
    opt.lambda_L1 = custom_args.get('lambda_L1', 100.0)
    opt.lambda_perceptual = custom_args.get('lambda_perceptual', 1.0)
    opt.use_4channel_dataset = custom_args.get('use_4channel_dataset', True)
    opt.mixed_precision = custom_args.get('mixed_precision', True)
    opt.gradient_checkpointing = custom_args.get('gradient_checkpointing', True)
    opt.use_perceptual_loss = custom_args.get('use_perceptual_loss', False)
    
    # Add missing options with defaults
    if not hasattr(opt, 'print_freq'):
        opt.print_freq = 100
    
    # Override some options for 4-channel high-res training
    opt.input_nc = 4   # 4-channel input
    opt.output_nc = 1  # 1-channel output
    opt.dataset_mode = 'four_channel'  # Use our 4-channel dataset loader
    
    # High-resolution specific settings
    opt.fineSize = 256  # Match the H,W dimensions of (4,64,256,256)
    opt.loadSize = 256  # Same as fineSize for high-res
    
    # Adjust batch size for high-resolution training (use smaller batches)
    if opt.batchSize > 2:
        print(f"⚠️ Large batch size ({opt.batchSize}) detected for high-res training")
        print(f"   Consider reducing to 1-2 for memory efficiency")
    
    # Set default 4-channel high-res architecture
    if not hasattr(opt, 'which_model_netG') or opt.which_model_netG in ['unet_128', 'unet_256']:
        opt.which_model_netG = 'unet_4channel_256'  # Use 256 architecture for high-res
        print(f"🔧 Using high-res 4-channel architecture: {opt.which_model_netG}")
    elif opt.which_model_netG == 'unet_4channel_128':
        opt.which_model_netG = 'unet_4channel_256'  # Upgrade to 256 for high-res
        print(f"🔧 Upgraded to high-res architecture: {opt.which_model_netG}")
    elif opt.which_model_netG == 'resnet_9blocks':
        opt.which_model_netG = 'resnet_4channel_9blocks'
        print(f"🔧 Using 4-channel ResNet architecture: {opt.which_model_netG}")
    
    print(f"🔧 High-Resolution 4-Channel Training Configuration:")
    print(f"   📊 Input channels: {opt.input_nc}")
    print(f"   📊 Output channels: {opt.output_nc}")
    print(f"   📊 Expected input shape: (4, 64, 256, 256)")
    print(f"   📊 Expected output shape: (1, 64, 256, 256)")
    print(f"   🏗️ Generator: {opt.which_model_netG}")
    print(f"   📁 Dataset mode: {opt.dataset_mode}")
    print(f"   ⚖️ L1 loss weight: {opt.lambda_L1}")
    print(f"   🎨 Perceptual loss weight: {opt.lambda_perceptual}")
    print(f"   🔧 Mixed precision: {opt.mixed_precision}")
    print(f"   💾 Gradient checkpointing: {opt.gradient_checkpointing}")
    print(f"   📏 Image size: {opt.fineSize}")
    
    # Create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'📊 Dataset size: {dataset_size}')
    
    # Create model
    print("🏗️ Creating high-resolution 4-channel model...")
    model = create_4channel_model(opt)
    
    # Training parameters
    total_steps = 0
    start_epoch = opt.epoch_count
    
    print(f"🚀 Starting high-resolution 4-channel training...")
    print(f"   Epochs: {start_epoch} to {opt.niter + opt.niter_decay}")
    print(f"   Batch size: {opt.batchSize}")
    print(f"   Learning rate: {opt.lr}")
    print(f"   Input shape: (4, 64, 256, 256)")
    print(f"   Output shape: (1, 64, 256, 256)")
    print(f"   Architecture: {opt.which_model_netG}")
    
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
                'G_perc': f"{errors['G_perceptual']:.3f}",
                'D_real': f"{errors['D_real']:.3f}",
                'D_fake': f"{errors['D_fake']:.3f}"
            })
            
            # Print losses
            if total_steps % opt.print_freq == 0:
                tqdm.write(f'📊 Step {total_steps}, Epoch {epoch}[{epoch_iter}/{dataset_size}]: ' + 
                          ', '.join([f'{k}: {v:.4f}' for k, v in errors.items()]))
        
        # Save model
        if epoch % opt.save_epoch_freq == 0:
            print(f'💾 Saving high-resolution model at epoch {epoch}')
            model.save('latest')
            model.save(f'epoch_{epoch}')
        
        # Update learning rates
        model.update_learning_rate()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'⏱️ High-res Epoch {epoch} completed in {epoch_time:.2f}s')

if __name__ == '__main__':
    main()