#!/bin/bash
# Fix Intel MKL threading issues
# Source this script before running PyTorch/NumPy operations

echo "🔧 Applying Intel MKL fixes..."

# Set threading layer to GNU instead of Intel
export MKL_THREADING_LAYER=GNU

# Force Intel MKL service
export MKL_SERVICE_FORCE_INTEL=1

# Limit number of threads to prevent conflicts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Alternative fix: Use MKL interface layer
# export MKL_INTERFACE_LAYER=GNU,LP64

echo "✅ Intel MKL environment configured"
echo "   MKL_THREADING_LAYER: $MKL_THREADING_LAYER"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS: $MKL_NUM_THREADS"

# Usage:
# source fix_mkl.sh
# python your_script.py
