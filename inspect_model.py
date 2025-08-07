#!/usr/bin/env python3
"""
Model Architecture Inspector
Loads a saved checkpoint and prints detailed model architecture information
"""

import os
import sys

# Fix MKL issues before importing torch
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

def inspect_model_architecture(checkpoint_path):
    """Load and inspect model architecture from checkpoint"""
    
    print(f"🔍 Inspecting model: {checkpoint_path}")
    print("=" * 80)
    
    # Load the checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {len(state_dict)}")
    print(f"   Checkpoint file size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    
    # Analyze layer structure
    print(f"\n🏗️ Layer Architecture:")
    print("-" * 80)
    
    for i, (layer_name, tensor) in enumerate(state_dict.items()):
        shape = tensor.shape
        param_count = tensor.numel()
        
        print(f"{i+1:3d}. {layer_name:<50} {str(shape):<25} ({param_count:,} params)")
        
        # Highlight important layers
        if 'output_layer' in layer_name and 'weight' in layer_name:
            print(f"     🎯 OUTPUT LAYER: {shape} -> Output channels: {shape[0]}")
        elif 'input' in layer_name.lower() and 'weight' in layer_name:
            if len(shape) >= 2:
                print(f"     📥 INPUT LAYER: {shape} -> Input channels: {shape[1]}")
    
    # Extract key architecture info
    print(f"\n🔧 Key Architecture Details:")
    print("-" * 80)
    
    # Find input channels
    input_channels = None
    for layer_name, tensor in state_dict.items():
        if ('conv' in layer_name.lower() or 'input' in layer_name.lower()) and 'weight' in layer_name:
            if len(tensor.shape) >= 2:
                input_channels = tensor.shape[1]
                print(f"   📥 Input channels: {input_channels}")
                break
    
    # Find output channels
    output_channels = None
    for layer_name, tensor in state_dict.items():
        if 'output_layer' in layer_name and 'weight' in layer_name:
            output_channels = tensor.shape[0]
            print(f"   📤 Output channels: {output_channels}")
            break
    
    # Detect architecture type
    print(f"\n🏛️ Architecture Type Detection:")
    print("-" * 80)
    
    layer_names = list(state_dict.keys())
    
    if any('unet_core' in name for name in layer_names):
        print("   🔹 Architecture: UNet-based")
        if any('model.3.model.3.model.3.model.3.model.3' in name for name in layer_names):
            print("   🔹 Depth: UNet-256 (5 levels)")
        elif any('model.3.model.3.model.3.model.3' in name for name in layer_names):
            print("   🔹 Depth: UNet-128 (4 levels)")
        else:
            print("   🔹 Depth: Custom UNet")
    elif any('resnet' in name.lower() for name in layer_names):
        print("   🔹 Architecture: ResNet-based")
    else:
        print("   🔹 Architecture: Unknown/Custom")
    
    # Check for 4-channel specific patterns
    if input_channels == 4:
        print("   🔹 Multi-channel input detected (4 channels)")
    if output_channels == 1:
        print("   🔹 Single-channel output (likely CT)")
    elif output_channels == 4:
        print("   🔹 Multi-channel output (latent-to-latent)")
    
    # Summary
    print(f"\n📋 Summary:")
    print("-" * 80)
    print(f"   Model Type: {input_channels}-channel → {output_channels}-channel")
    if input_channels == 4 and output_channels == 1:
        print("   Task: Latent MR → Synthetic CT")
    elif input_channels == 4 and output_channels == 4:
        print("   Task: Latent-to-Latent Translation")
    else:
        print(f"   Task: {input_channels}→{output_channels} channel translation")

def main():
    # Default checkpoint path
    checkpoint_dir = "/home/sawall/scratch/checkpoints/latent_to_latent"
    checkpoint_path = os.path.join(checkpoint_dir, "netG_latest.pth")
    
    # Allow custom path as command line argument
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\n💡 Usage:")
        print(f"   python inspect_model.py [checkpoint_path]")
        print(f"\n   Example:")
        print(f"   python inspect_model.py /home/sawall/scratch/checkpoints/latent_to_latent/netG_latest.pth")
        return
    
    # Inspect the model
    inspect_model_architecture(checkpoint_path)

if __name__ == '__main__':
    main()
