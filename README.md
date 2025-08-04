# Eric's Camera library

## Work-in-progress! Only works for the exact workflows I use right now.

Provides Python classes to interact with various cameras. So far there is support for
- CSI cameras on Jetson Orin platforms
- USB cameras on Linux platforms
- RTSP cameras on Linux platforms
- Arbitrary gstreamer pipelines

The package also includes a nice calibration utility that can be run with `python3 -m erics_cameras.calibrate`, that will automatically select which images to include in the calibration dataset by criteria like stillness, uniqueness of the camera pose, and reprojection error. It will also show you the reprojections of the calibration target points using the latest calculated intrinsics in real time, and the intrinsics will be printed to the terminal for you to copy into your program with numpy.