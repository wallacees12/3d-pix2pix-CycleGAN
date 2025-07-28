# Latent MR Testing Pipeline

This folder contains scripts for testing 4-channel latent MR representations through trained models and generating synthetic CT with proper upscaling.

## 🎯 **Workflow Overview**

1. **Prepare Test Data** - Create crops.pkl from latent MR files with downsampling information
2. **Run Inference** - Process latent MR through trained model to generate synthetic CT  
3. **Upscale Results** - Use stored transformation info to properly upscale synthetic CT

## 📁 **Files**

### `create_test_crops.py`
Creates `crops.pkl` file compatible with `launch_4channel_training.py` for testing phase.

**Features:**
- Finds 4-channel latent MR files (shape: 4,32,128,128 or similar)
- Extracts downsampling transformation info by comparing with original MR files
- Stores scale factors, spacing, and shape information for proper upscaling
- Compatible with existing training infrastructure

**Usage:**
```bash
# List available files
python create_test_crops.py /path/to/test/data --list-files

# Create crops.pkl
python create_test_crops.py /path/to/test/data
```

**Expected data structure:**
```
test_data/
├── patient_001_mr_all_channels.npz    # 4-channel latent MR
├── patient_001_mr.mha                 # Original high-res MR (for shape info)
├── patient_001_ct.mha                 # Original CT (preferred for spacing info)
├── patient_002_mr_all_channels.npz
├── patient_002_mr.mha
├── patient_002_ct.mha
└── ... 
```

**Note:** If both MR and CT files are available, the script will use:
- **CT spacing/orientation** (preferred for target domain accuracy)
- **MR shape information** (for transformation calculation)
- This gives the most accurate reconstruction parameters

### `launch_test_4channel.py`
Launches testing using existing `launch_4channel_training.py` infrastructure.

**Features:**
- Validates data structure before testing
- Runs inference on latent MR representations
- Uses trained models to generate synthetic CT
- Compatible with existing checkpoint format

**Usage:**
```bash
# Check data structure
python launch_test_4channel.py --dataroot /path/to/test/data --name your_model --check_data

# Run testing
python launch_test_4channel.py --dataroot /path/to/test/data --name your_model --which_epoch latest
```

### `upscale_with_transform_info.py`
Upscales synthetic CT using transformation information from training data.

**Features:**
- Loads downsampling info from crops.pkl
- Reverses exact transformation used during training
- Supports both scipy (training-consistent) and SimpleITK (medical-grade) methods
- Preserves medical imaging spacing and orientation

**Usage:**
```bash
# Upscale using stored transformation info
python upscale_with_transform_info.py --npz_file results/model/test_latest/patient_001.npz --crops_pkl /path/to/test/data/crops.pkl

# Use SimpleITK for medical-grade resampling
python upscale_with_transform_info.py --npz_file results/model/test_latest/patient_001.npz --crops_pkl /path/to/test/data/crops.pkl --method sitk
```

## 🚀 **Complete Workflow Example**

```bash
# 1. Prepare test data
cd /Users/samwallace/Documents/sCT/Comparison\ Pix2Pix/3D/new
python create_test_crops.py /path/to/latent/mr/data

# 2. Run inference 
python launch_test_4channel.py --dataroot /path/to/latent/mr/data --name 4channel_mr_to_ct --check_data
python launch_test_4channel.py --dataroot /path/to/latent/mr/data --name 4channel_mr_to_ct

# 3. Upscale results
python upscale_with_transform_info.py --npz_file ../results/4channel_mr_to_ct/test_latest/patient_001.npz --crops_pkl /path/to/latent/mr/data/crops.pkl
```

## 🔧 **Key Features**

### **Downsampling Information Extraction**
- Compares original high-res MR with latent representation
- Calculates exact scale factors used in training
- Stores spacing, origin, and orientation for proper reconstruction

### **Training Pipeline Integration**  
- Uses existing `launch_4channel_training.py` with `--phase test`
- Compatible with trained model checkpoints
- Maintains data loading format and augmentation settings

### **Proper Medical Upscaling**
- Scipy method: Same as training pipeline for consistency
- SimpleITK method: Medical-grade resampling for anatomical accuracy
- Preserves spatial relationships and HU values

## 📊 **Data Flow**

```
Original MR (256³) → Latent MR (4×32×128×128) → Synthetic CT (32×128×128) → Upscaled CT (256³)
     ↓                        ↓                         ↓                    ↑
Extract transform    →    Store in crops.pkl    →    Use for upscaling ----┘
```

## ⚠️ **Requirements**

- Latent MR files in NPZ format with 4 channels
- Original MR files in MHA format (for transformation extraction)
- Trained 4-channel model checkpoints
- Python packages: numpy, SimpleITK, scipy, pickle

## 💡 **Tips**

1. **File Naming**: Use consistent naming between latent and original files:
   - `patient_001_latent.npz` ↔ `patient_001.mha`
   - `patient_001_mr_latent.npz` ↔ `patient_001_mr.mha`

2. **Testing**: Always use `--check_data` first to validate data structure

3. **Upscaling Method**: 
   - Use `scipy` for consistency with training
   - Use `sitk` for medical imaging accuracy

4. **Batch Processing**: Scripts support multiple samples automatically
