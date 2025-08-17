# Fisheye Camera Calibration Implementation

This document describes the implementation of fisheye camera calibration using the `cv2.fisheye` module, replacing the standard OpenCV calibration functions.

## Overview

The original calibration code used standard OpenCV functions (`cv2.calibrateCamera`, `cv2.undistort`, etc.) which are designed for pinhole cameras with radial and tangential distortion. For fisheye cameras, we now use the specialized `cv2.fisheye` module which provides:

- Fisheye-specific calibration (`cv2.fisheye.calibrate`)
- Fisheye point projection (`cv2.fisheye.projectPoints`)
- Fisheye undistortion (`cv2.fisheye.undistortImage`)
- Fisheye undistortion maps (`cv2.fisheye.initUndistortRectifyMap`)

## Key Changes Made

### 1. Distortion Model

**Before (Standard OpenCV):**
```python
# 5x1 distortion coefficients: [k1, k2, p1, p2, k3]
# Radial distortion: r_distorted = r_undistorted * (1 + k1*r² + k2*r⁴ + k3*r⁶)
# Tangential distortion: p1, p2
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
```

**After (Fisheye):**
```python
# 4x1 distortion coefficients: [k1, k2, k3, k4]
# Fisheye distortion: r_distorted = r_undistorted * (1 + k1*r + k2*r² + k3*r³ + k4*r⁴)
# No tangential distortion for fisheye cameras
dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32).reshape(4, 1)
```

### 2. Distortion Coefficient Generation

**Before:** Complex monotonicity constraints for radial distortion
**After:** Fisheye-specific coefficient generation with appropriate ranges:

```python
def generate_fisheye_distortion_coeffs():
    # k1: Strong negative value for barrel distortion (fisheye effect)
    k1 = np.random.uniform(-0.8, -0.2)
    
    # k2: Positive value to compensate and maintain monotonicity
    k2 = np.random.uniform(0.1, 0.6)
    
    # k3, k4: Smaller values for fine-tuning
    k3 = np.random.uniform(-0.1, 0.1)
    k4 = np.random.uniform(-0.05, 0.05)
    
    return k1, k2, k3, k4
```

### 3. Undistortion Maps

**Before:**
```python
new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, resolution, 1, resolution)
inv_distort_maps = cv2.initInverseRectificationMap(cam_matrix, dist_coeffs, None, new_cam_mat, resolution, cv2.CV_16SC2)
```

**After:**
```python
new_K = cam_matrix.copy()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    cam_matrix, dist_coeffs, None, new_K, resolution, cv2.CV_16SC2
)
inv_distort_maps = (map1, map2)
```

**Note:** Fisheye undistortion maps have different shapes:
- `map1`: (height, width, 2) - int16
- `map2`: (height, width) - uint16

### 4. Point Projection

**Before:**
```python
grid_proj_inaccurate = cv2.projectPoints(points_3d, rvec, tvec, self.cam_matrix, self.distortion_coeffs)[0].squeeze()
```

**After:**
```python
grid_proj_inaccurate = cv2.fisheye.projectPoints(
    points_3d.reshape(-1, 1, 3), 
    rvec.reshape(3, 1), 
    tvec.reshape(3, 1), 
    self.cam_matrix, 
    self.distortion_coeffs
)[0].squeeze()
```

## Files Modified

### 1. `calib_data_gen.py`

- **Replaced** `generate_monotonic_distortion_coeffs()` with `generate_fisheye_distortion_coeffs()`
- **Updated** distortion coefficient generation to use fisheye model (4x1 instead of 5x1)
- **Modified** `make_datapoint()` to use fisheye undistortion maps
- **Added** `test_fisheye_distortion()` function for testing

### 2. `opencv_render.py`

- **Updated** `OpenCVRenderer.render_image()` to use fisheye undistortion maps
- **Modified** `get_keypoint_labels()` to use fisheye point projection
- **Updated** documentation to reflect fisheye-specific parameters

### 3. New Files Created

- **`test_fisheye.py`**: Comprehensive test suite for fisheye functionality
- **`demo_fisheye.py`**: Demonstration script showing fisheye effects
- **`FISHEYE_IMPLEMENTATION.md`**: This documentation file

## Testing

The implementation includes comprehensive testing:

```bash
# Run all tests
python test_fisheye.py

# Run demonstration
python demo_fisheye.py
```

### Test Coverage

1. **Fisheye distortion coefficient generation**
2. **Camera matrix generation with fisheye distortion**
3. **Fisheye undistortion map generation**
4. **Fisheye point projection**
5. **OpenCVRenderer fisheye rendering**

## Fisheye Distortion Characteristics

The implemented fisheye model produces:

- **Barrel distortion**: Negative k1 values create the characteristic fisheye "bulge"
- **Compensation**: Positive k2 values prevent extreme distortion
- **Fine-tuning**: Small k3, k4 values for subtle adjustments
- **Monotonicity**: Ensured through careful coefficient ranges

## Example Usage

```python
from calib_data_gen import generate_fisheye_distortion_coeffs
from opencv_render import OpenCVRenderer

# Generate fisheye distortion coefficients
k1, k2, k3, k4 = generate_fisheye_distortion_coeffs()
dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32).reshape(4, 1)

# Create renderer with fisheye parameters
renderer = OpenCVRenderer(cam_matrix, dist_coeffs)

# Render with fisheye distortion
img = renderer.render_image((width, height), rvec, tvec)
```

## Benefits of Fisheye Implementation

1. **Accurate modeling**: Fisheye-specific distortion model matches real fisheye cameras
2. **Better calibration**: Specialized algorithms for fisheye lens characteristics
3. **Realistic rendering**: More accurate simulation of fisheye camera behavior
4. **Proper undistortion**: Correct handling of fisheye distortion patterns
5. **Performance**: Optimized algorithms for fisheye camera operations

## Compatibility

- **OpenCV Version**: Requires OpenCV 3.0+ with contrib modules
- **Python**: Compatible with Python 3.6+
- **Dependencies**: numpy, opencv-contrib-python

## Future Enhancements

1. **Advanced fisheye models**: Support for different fisheye projection models
2. **Calibration tools**: Integration with fisheye calibration workflows
3. **Performance optimization**: GPU acceleration for large-scale rendering
4. **Validation tools**: Metrics for fisheye distortion quality assessment
