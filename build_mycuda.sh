#!/bin/bash
set -e

ROOT=$(pwd)

export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"

cd ${ROOT}/mycuda
rm -rf build *.egg-info *.so

python -m pip install -e . --no-deps

# Verify
python -c "import common; import gridencoder; print('mycuda modules loaded successfully')"

echo "✓ mycuda build complete!"

# pip uninstall common -y && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121