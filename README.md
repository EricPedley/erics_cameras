# Eric's Camera library
<img width="1736" height="954" alt="image" src="https://github.com/user-attachments/assets/60ced5f0-972f-42f0-823b-60402e7acb74" />

## Work-in-progress! Only works for the exact workflows I use right now.

Provides Python classes to interact with various cameras. So far there is support for
- CSI cameras on Jetson Orin platforms
- USB cameras on Linux platforms
- RTSP cameras on Linux platforms
- Arbitrary gstreamer pipelines

The package also includes a nice calibration utility that can be run with `python3 -m erics_cameras.calibrate`, that will automatically select which images to include in the calibration dataset by criteria like stillness, uniqueness of the camera pose, and reprojection error. It will also show you the reprojections of the calibration target points using the latest calculated intrinsics in real time, and the intrinsics will be printed to the terminal for you to copy into your program with numpy.

This package requires you have a version of OpenCV with GStreamer support. The repo includes a `build_opencv.sh` script that will download the opencv 4.12 source code to `~/Downloads` and install it with gstreamer support. You can test this is successfull with `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep 
GStreamer`. The calibration script will break with opencv versions less than 4.10 and if you don't have the opencv contrib package installed. If another package requires opencv and pip automatically installs it, it will override the version built from source. To make this package work again you will need to `pip uninstall opencv-python`, which only removes the version installed with pip, and leaves the system opencv working.

I've tested this package to work with Ubuntu 22 on an x86 pc and Jetson Orin Nano and NX. If you're doing robotics in 2025 I think this is the happy path in terms of OS and hardware for getting max compatibility with other programs like Isaac Sim, ROS, and Gazebo.
