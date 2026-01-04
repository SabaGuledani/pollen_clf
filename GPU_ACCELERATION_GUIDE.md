# GPU Acceleration Guide for LBP+HOG SVM Training

This guide explains how to enable GPU acceleration for the SVM classifier training using cuML.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (Compute Capability 6.0 or higher)
2. **CUDA Toolkit** installed (version 11.0 or 12.0)
3. **NVIDIA GPU Drivers** installed

## Check Your System

First, verify you have an NVIDIA GPU:

```bash
nvidia-smi
```

If this command works and shows your GPU, you're good to go!

## Installation Methods

### Method 1: Using Conda (Recommended - Easiest)

Conda is the easiest way to install cuML as it handles all dependencies automatically:

```bash
# Create a new conda environment (recommended)
conda create -n rapids-env python=3.10
conda activate rapids-env

# Install cuML (for CUDA 11.8)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.04 python=3.10 cudatoolkit=11.8

# Or for CUDA 12.0
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.10 cudatoolkit=12.7
```

**Note:** Replace `24.04` with the latest version available. Check [RAPIDS installation guide](https://rapids.ai/install) for the latest version.

### Method 2: Using pip (Alternative)

**Warning:** pip installation can be tricky due to CUDA dependencies. Only use if conda doesn't work.

```bash
# For CUDA 11.x
pip install cuml-cu11

# For CUDA 12.x
pip install cuml-cu12
```

You may also need to install CUDA toolkit separately:
- Download from: https://developer.nvidia.com/cuda-downloads
- Follow installation instructions for your OS

### Method 3: Docker (Most Reliable)

If you have Docker installed, this is the most reliable method:

```bash
# Pull RAPIDS container
docker pull rapidsai/rapidsai:24.04-cuda11.8-runtime-ubuntu22.04-py3.10

# Run container with GPU access
docker run --gpus all --rm -it \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:24.04-cuda11.8-runtime-ubuntu22.04-py3.10
```

## Verify Installation

After installation, verify cuML works:

```python
import cuml
print(f"cuML version: {cuml.__version__}")

# Check GPU
from cuml.common import device_type_from_context
print(f"GPU available: {cuml.is_available()}")
```

Or run the script - it will automatically check:

```bash
python train_lbp_hog_svm.py
```

The script will show:
- ✓ NVIDIA GPU detected
- ✓ cuML is installed and available

## Using GPU Acceleration

### Option 1: Automatic (Recommended)

The script automatically detects and uses GPU if available:

```python
from train_lbp_hog_svm import train_classifier

results = train_classifier(
    dataset_path="./data/data/train",
    # ... other parameters ...
    use_gpu=True,  # Will use GPU if cuML is available, otherwise falls back to CPU
)
```

### Option 2: Manual Check

```python
from train_lbp_hog_svm import train_classifier, check_gpu_availability

# Check if GPU is available
gpu_available = check_gpu_availability()

results = train_classifier(
    dataset_path="./data/data/train",
    # ... other parameters ...
    use_gpu=gpu_available,  # Use GPU only if available
)
```

## Performance Expectations

GPU acceleration primarily speeds up:
- **SVM training** - Can be 10-100x faster for large datasets
- **SVM prediction** - Faster inference

**Note:** Feature extraction (LBP+HOG) still runs on CPU with multiprocessing. The GPU only accelerates the SVM training step.

## Troubleshooting

### Issue: "cuML not available"
- **Solution:** Install cuML using one of the methods above
- Check CUDA version: `nvcc --version`
- Ensure GPU drivers are installed: `nvidia-smi`

### Issue: "CUDA out of memory"
- **Solution:** Reduce batch size or use smaller dataset
- Close other GPU applications
- Use CPU instead: `use_gpu=False`

### Issue: Import errors
- **Solution:** Ensure all dependencies are installed
- Try creating a fresh conda environment
- Check Python version compatibility (3.8-3.11 recommended)

### Issue: Slow performance
- **Solution:** 
  - Verify GPU is being used: Check `nvidia-smi` during training
  - For small datasets, CPU might be faster due to overhead
  - Ensure data is on GPU (cuML handles this automatically)

## Alternative: CPU-Only Optimization

If GPU is not available, the code is already optimized with:
- **Vectorized LBP computation** (10-100x faster than loops)
- **Multiprocessing** for parallel feature extraction
- **Efficient NumPy operations**

For most use cases, CPU with multiprocessing is fast enough!

## Resources

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [RAPIDS Installation Guide](https://rapids.ai/install)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)

## Quick Start Example

```python
from train_lbp_hog_svm import train_classifier, check_gpu_availability

# Check GPU
gpu_available = check_gpu_availability()

# Train with GPU if available
results = train_classifier(
    dataset_path="./data/data/train",
    train_set="train",
    valid_set="valid",
    lbp_radius=1.5,
    lbp_neighbors=6,
    lbp_grid_rows=2,
    lbp_grid_cols=2,
    hog_block_size=32,
    hog_block_stride=16,
    hog_cell_size=16,
    svm_kernel=2,  # RBF
    svm_C=1.0,
    svm_gamma=0.1,
    use_gpu=gpu_available,  # Automatically use GPU if available
    model_fname="model_gpu.pkl",
    n_jobs=-1  # Use all CPUs for feature extraction
)

print(f"Training accuracy: {results['Train_Accuracy']:.6f}")
```

