# Installing cuML on Windows

## Problem
cuML doesn't have Windows wheels available via pip. The pip packages only work on Linux.

## Solution Options

### Option 1: Use Conda (Recommended for Windows)

Conda has better Windows support for RAPIDS/cuML:

```bash
# Activate your conda environment
conda activate AI

# Install cuML via conda
conda install -c rapidsai -c conda-forge -c nvidia cuml python=3.9 cudatoolkit=12.0

# Or try with mamba (faster)
conda install mamba -c conda-forge
mamba install -c rapidsai -c conda-forge -c nvidia cuml python=3.9 cudatoolkit=12.0
```

**Note:** You're using Python 3.9.21, so make sure to specify `python=3.9` in the conda command.

### Option 2: Use WSL (Windows Subsystem for Linux)

If conda doesn't work, you can use WSL2 with Linux:

1. Install WSL2: `wsl --install`
2. Install Ubuntu from Microsoft Store
3. Install CUDA in WSL: https://docs.nvidia.com/cuda/wsl-user-guide/
4. Install cuML in WSL using pip (Linux wheels will work)

### Option 3: Use CPU Version (Easiest - Already Optimized!)

**Good news:** The code is already highly optimized for CPU:
- ✅ Vectorized LBP computation (10-100x faster)
- ✅ Multiprocessing for parallel feature extraction
- ✅ Efficient NumPy operations

For most datasets, CPU with multiprocessing is fast enough! GPU only helps with very large SVM training.

**Just use:**
```python
results = train_classifier(
    # ... your parameters ...
    use_gpu=False,  # Use optimized CPU version
    n_jobs=-1      # Use all CPU cores
)
```

## Try Conda Installation

Run this in your conda environment:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml python=3.9 cudatoolkit=12.0
```

If that fails, try with a specific version:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.9 cudatoolkit=12.0
```

## Verify Installation

After installation, test:

```python
import cuml
print(f"cuML version: {cuml.__version__}")
print(f"GPU available: {cuml.is_available()}")
```

## Recommendation

For your use case (pollen classification), the **CPU version with multiprocessing is likely sufficient** and will be much easier to set up. GPU acceleration mainly helps with:
- Very large datasets (100k+ samples)
- Complex models with many features

Your current setup with vectorized operations and multiprocessing should be fast enough!

