#!/bin/bash

set -e  # Exit on error

ENV_NAME="bundlesdf"
BUILD_DIR="/tmp/bundlesdf_build"

echo "================================"
echo "BundleSDF Environment Setup"
echo "================================"

# Clean up any existing build directory
if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing build directory..."
    rm -rf ${BUILD_DIR}
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting. Please manually remove the environment or use a different name."
        exit 1
    fi
fi

echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Installing CUDA toolkit 12.4 (consistent version)..."
conda install -c "nvidia/label/cuda-12.4.0" \
    cuda-nvcc=12.4.99 \
    cuda-cudart=12.4.99 \
    cuda-cudart-dev=12.4.99 \
    cuda-cudart-static=12.4.99 \
    cuda-nvrtc=12.4.127 \
    cuda-nvrtc-dev=12.4.127 \
    cuda-libraries-dev \
    npp -y

echo "Installing cuDNN..."
conda install -c conda-forge cudnn -y

echo "Installing build dependencies via conda..."
conda install -c conda-forge \
    cmake=3.25.3 \
    gcc=11.4.0 gxx=11.4.0 gfortran=11.4.0 \
    make ninja pkg-config \
    git wget curl bzip2 \
    boost-cpp \
    flann \
    tbb-devel \
    glog gflags \
    hdf5 \
    libjpeg-turbo libtiff libpng freetype \
    freeglut glew \
    ffmpeg x264 \
    yasm \
    openssl \
    zeromq \
    cppzmq \
    openblas libblas liblapack \
    libgomp openmp \
    pcl=1.10.0 \
    pybind11=2.13.0 \
    yaml-cpp=0.8.0 \
    "libprotobuf<3.20" "protobuf<3.20" -y

# Create build directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Set build variables
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# CRITICAL: targets/include FIRST for npp.h, then conda includes
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include

# System lib paths FIRST for glibc symbols (avoids undefined reference errors)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH

export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1

echo "================================"
echo "Building Eigen 3.4.0..."
echo "================================"
cd ${BUILD_DIR}
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make install
cd ${BUILD_DIR}
rm -rf eigen-3.4.0*

echo "================================"
echo "Building OpenCV 4.11.0 with CUDA..."
echo "================================"
cd ${BUILD_DIR}
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv_contrib
mkdir -p opencv/build && cd opencv/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_LIBDIR=lib \
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
    -DINSTALL_PKGCONFIG=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DPYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.10 \
    -DPYTHON3_LIBRARY=$CONDA_PREFIX/lib/libpython3.10.so

make -j$(nproc)
make install
cd ${BUILD_DIR}
rm -rf opencv*

# Clean up build directory
cd ~
rm -rf ${BUILD_DIR}

echo "================================"
echo "Installing Python packages..."
echo "================================"

echo "Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing kaolin..."
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

echo "Installing PyTorch3D..."
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu121

echo "Installing remaining Python packages..."
pip install trimesh wandb matplotlib imageio tqdm open3d ruamel.yaml sacred \
    kornia pymongo pyrender jupyterlab ninja "Cython>=0.29.37" yacs scipy \
    scikit-learn numpy==1.26.4 transformations einops scikit-image \
    awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui \
    pytinyrenderer PyQt5 cython-npm chardet openpyxl

echo "Downloading freeimage binary..."
python -c "import imageio; imageio.plugins.freeimage.download()"

echo "================================"
echo "Setting up environment variables..."
echo "================================"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Create activation script
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/sh
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

# CRITICAL: targets/include first for CUDA headers (npp.h)
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include

# System libs first for glibc symbols
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib

export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
export DISPLAY=${DISPLAY:-:0}
EOF

# Create deactivation script
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/sh
unset CC
unset CXX
unset CUDACXX
unset CUDA_HOME
unset CUDA_PATH
unset OPENCV_IO_ENABLE_OPENEXR
unset PYTHONUNBUFFERED
unset OMP_NUM_THREADS
unset CMAKE_PREFIX_PATH
unset CPATH
unset CPLUS_INCLUDE_PATH
unset TORCH_CUDA_ARCH_LIST
unset FORCE_CUDA
unset TORCH_EXTENSIONS_DIR
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "IMPORTANT: Reactivate the environment to load new variables:"
echo "  conda deactivate"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo "  python -c 'import cv2; print(f\"OpenCV: {cv2.__version__}, CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}\")'"
echo ""
echo "Build BundleSDF components with:"
echo "  ./build_mycuda.sh"
echo "  ./build_bundletrack.sh"
echo ""
echo "Note: This build process may take 1-2 hours depending on your system."
echo ""
