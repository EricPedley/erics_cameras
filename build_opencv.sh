cd ~/Downloads
wget -O opencv-4.12.0.zip https://github.com/opencv/opencv/archive/4.12.0.zip
wget -O opencv_contrib-4.12.0.zip https://github.com/opencv/opencv_contrib/archive/4.12.0.zip

unzip opencv-4.12.0.zip
unzip opencv_contrib-4.12.0.zip

cd opencv-4.12.0
mkdir build && cd build

PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PYTHON_EXEC=$(which python3)

PREFIX=/usr/local
# if conda prefix, use that
if [[ $CONDA_PREFIX != "" ]]; then
    PREFIX=$CONDA_PREFIX
fi

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$PREFIX \
    -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-4.12.0/modules \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D BUILD_PYTHON_BINDINGS=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D PYTHON3_EXECUTABLE=$PYTHON_EXEC \
    -D PYTHON3_INCLUDE_DIR=$PYTHON_INCLUDE \
    -D PYTHON3_PACKAGES_PATH=$PYTHON_PACKAGES \
    ..

make -j$(nproc)
# If you get shit about arm_neon.h no such file, run commands roughly like this (change the specifics):
# export CC=/home/dpsh/miniconda3/envs/vr/bin/aarch64-conda-linux-gnu-gcc
# export CXX=/home/dpsh/miniconda3/envs/vr/bin/aarch64-conda-linux-gnu-g++
# export CFLAGS="-I/home/dpsh/miniconda3/envs/vr/lib/gcc/aarch64-conda-linux-gnu/10.4.0/include"
# export CXXFLAGS="-I/home/dpsh/miniconda3/envs/vr/lib/gcc/aarch64-conda-linux-gnu/10.4.0/include"
sudo make install
sudo ldconfig