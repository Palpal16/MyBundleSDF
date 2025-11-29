#!/bin/bash
set -e

export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include

# System libs first for glibc symbols
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1

cd /tmp
rm -rf opencv opencv_contrib

git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv_contrib

mkdir -p opencv/build && cd opencv/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_CUDA_STUBS=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DWITH_MATLAB=OFF \
    -DCUDA_FAST_MATH=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_OPENMP=ON \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DWITH_OPENEXR=OFF \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DBUILD_opencv_cudacodec=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_FFMPEG=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_opencv_xfeatures2d=OFF \
    -DOPENCV_DNN_OPENCL=OFF \
    -DWITH_CUDA=ON \
    -DWITH_OPENCL=OFF \
    -DBUILD_opencv_wechat_qrcode=OFF \
    -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0" \
    -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DINSTALL_PKGCONFIG=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF

make -j$(nproc)
make install

cd /tmp && rm -rf opencv opencv_contrib

echo "✓ OpenCV build complete!"