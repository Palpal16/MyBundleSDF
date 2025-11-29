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

cd ${ROOT}/BundleTrack
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_C_FLAGS="-fopenmp" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_C_LIB_NAMES="gomp;pthread" \
  -DOpenMP_CXX_LIB_NAMES="gomp;pthread" \
  -DOpenMP_gomp_LIBRARY="$CONDA_PREFIX/lib/libgomp.so" \
  -DOpenMP_pthread_LIBRARY="/usr/lib/x86_64-linux-gnu/libpthread.so.0" \
  -DBLAS_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so" \
  -DLAPACK_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so"

make -j11

echo "✓ BundleTrack build complete!"