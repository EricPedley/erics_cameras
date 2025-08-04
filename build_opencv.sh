cd ~/Downloads
wget -O opencv-4.12.0.zip https://github.com/opencv/opencv/archive/4.12.0.zip
wget -O opencv_contrib-4.12.0.zip https://github.com/opencv/opencv_contrib/archive/4.12.0.zip

unzip opencv-4.12.0.zip
unzip opencv_contrib-4.12.0.zip

cd opencv-4.12.0
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-4.12.0/modules \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D BUILD_PYTHON_BINDINGS=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.* \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.*/site-packages \
    ..

make -j$(nproc)
sudo make install
sudo ldconfig