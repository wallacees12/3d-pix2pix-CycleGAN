import os
import torch

BASE_DIR = "/home/sawall/scratch/checkpoints"  # Set your base directory here
MODEL_FILENAME = "netG_latest.pth"
MODEL_FILENAME2 = "latest_net_G.pth"

def analyze_checkpoint(checkpoint_path):
    print(f"\n🔍 Analyzing: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        first_conv_key, last_conv_key = None, None

        for key in state_dict:
            if 'weight' in key and len(state_dict[key].shape) >= 4:
                if first_conv_key is None:
                    first_conv_key = key
                last_conv_key = key

        print(f"   Total params: {len(state_dict)}")

        if any("input_adapter" in k for k in state_dict):
            print(f"   ✅ Found input_adapter")
        if any("output_layer" in k for k in state_dict):
            print(f"   ✅ Found output_layer")

        if first_conv_key:
            shape = state_dict[first_conv_key].shape
            print(f"   🔢 First conv: {first_conv_key} → {shape} → Input channels: {shape[1]}")
        if last_conv_key:
            shape = state_dict[last_conv_key].shape
            print(f"   🔢 Last conv: {last_conv_key} → {shape} → Output channels: {shape[0]}")

        print("   ✅ Done\n")

    except Exception as e:
        print(f"❌ Failed to analyze {checkpoint_path}: {e}")


def main():
    print(f"📁 Scanning for checkpoints in: {BASE_DIR}\n")
    folders = sorted(os.listdir(BASE_DIR))
    total_found = 0

    for folder in folders:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        checkpoint_path = os.path.join(folder_path, MODEL_FILENAME)
        if os.path.exists(checkpoint_path):
            analyze_checkpoint(checkpoint_path)
            total_found += 1
        elif os.path.exists(os.path.join(folder_path, MODEL_FILENAME2)):
            analyze_checkpoint(os.path.join(folder_path, MODEL_FILENAME2))
            total_found += 1
        else:
            print(f"⚠️ No checkpoint found in {folder}")

    print(f"\n✅ Checked {total_found} models.")


if __name__ == "__main__":
    main()
