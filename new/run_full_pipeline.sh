#!/bin/bash

# Complete testing pipeline for 4-channel latent MR to synthetic CT
# This script runs the full workflow from data preparation to upscaled results

set -e  # Exit on any error

# Configuration
DATA_DIR=""
MODEL_NAME=""
EPOCH="latest"
GPU_IDS="0"
METHOD="scipy"  # or "sitk"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --data_dir PATH --model_name NAME [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --data_dir PATH      Directory containing latent MR files and original files"
            echo "  --model_name NAME    Name of trained model"
            echo ""
            echo "Optional:"
            echo "  --epoch EPOCH        Model epoch to use (default: latest)"
            echo "  --gpu_ids IDS        GPU IDs to use (default: 0)"
            echo "  --method METHOD      Upscaling method: scipy or sitk (default: scipy)"
            echo ""
            echo "Note: Uses existing scripts/launch_4channel_testing.py for inference"
            echo ""
            echo "Example:"
            echo "  $0 --data_dir /path/to/test/data --model_name 4channel_mr_to_ct"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_DIR" ]]; then
    echo "❌ Error: --data_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "❌ Error: --model_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "🚀 4-Channel Latent MR Testing Pipeline"
echo "======================================"
echo "Data directory: $DATA_DIR"
echo "Model name: $MODEL_NAME"
echo "Epoch: $EPOCH"
echo "GPU IDs: $GPU_IDS"
echo "Upscaling method: $METHOD"
echo ""

# Step 1: Create crops.pkl
echo "📋 Step 1: Creating crops.pkl with downsampling information..."
python create_test_crops.py "$DATA_DIR"

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to create crops.pkl"
    exit 1
fi

echo "✅ crops.pkl created successfully"
echo ""

# Step 2: Run inference using existing testing script
echo "🧠 Step 2: Running inference through trained model..."
python ../scripts/launch_4channel_testing.py \
    --dataroot "$DATA_DIR" \
    --name "$MODEL_NAME" \
    --which_epoch "$EPOCH" \
    --gpu_ids "$GPU_IDS"

if [[ $? -ne 0 ]]; then
    echo "❌ Inference failed"
    exit 1
fi

echo "✅ Inference completed"
echo ""

# Step 3: Find result files and upscale
echo "📈 Step 3: Upscaling synthetic CT results..."

# The existing script saves to test_{epoch}_npz format
RESULTS_DIR="../results/$MODEL_NAME/test_${EPOCH}_npz"
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "⚠️  Results directory not found: $RESULTS_DIR"
    echo "Looking for alternative result locations..."
    
    # Try to find results directory
    RESULTS_BASE="../results/$MODEL_NAME"
    if [[ -d "$RESULTS_BASE" ]]; then
        RESULTS_DIR=$(find "$RESULTS_BASE" -name "test_*" -type d | head -1)
        if [[ -n "$RESULTS_DIR" ]]; then
            echo "✅ Found results in: $RESULTS_DIR"
        fi
    fi
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "❌ Could not find results directory"
    echo "Expected: $RESULTS_DIR"
    exit 1
fi

# Find NPZ files in results
NPZ_FILES=($(find "$RESULTS_DIR" -name "*.npz" -type f))

if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
    echo "❌ No NPZ result files found in $RESULTS_DIR"
    exit 1
fi

echo "📊 Found ${#NPZ_FILES[@]} result files to upscale"

# Upscale each result file
for npz_file in "${NPZ_FILES[@]}"; do
    echo "🔄 Upscaling: $(basename "$npz_file")"
    
    python upscale_with_transform_info.py \
        --npz_file "$npz_file" \
        --crops_pkl "$DATA_DIR/crops.pkl" \
        --method "$METHOD"
    
    if [[ $? -ne 0 ]]; then
        echo "⚠️  Failed to upscale: $(basename "$npz_file")"
    else
        echo "✅ Upscaled: $(basename "$npz_file")"
    fi
done

echo ""
echo "🎉 Pipeline completed successfully!"
echo ""
echo "📁 Results available in:"
echo "   - Synthetic CT: $RESULTS_DIR"
echo "   - Upscaled CT: $RESULTS_DIR/*_upscaled.*"
echo ""
echo "💡 Next steps:"
echo "   - Review upscaled CT quality"
echo "   - Compare with ground truth if available" 
echo "   - Adjust upscaling method if needed (--method sitk)"
