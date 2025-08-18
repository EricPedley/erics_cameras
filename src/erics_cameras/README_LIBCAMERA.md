# LibCameraCam - Raspberry Pi Camera with Exposure Control

The `LibCameraCam` class provides a clean interface for controlling Raspberry Pi cameras using libcamera through GStreamer, with comprehensive exposure and image quality controls.

## Features

- **Exposure Control**: Auto and manual exposure modes with precise timing control
- **Gain Control**: Auto and manual analogue/digital gain control
- **Image Quality**: Brightness, contrast, saturation, and sharpness adjustments
- **Auto-Exposure**: Advanced auto-exposure with EV adjustment
- **White Balance**: Auto white balance with manual mode selection
- **Autofocus**: Manual, auto, and continuous focus modes
- **Resolution Options**: Predefined resolutions (1080p, 720p, 480p) or custom
- **Real-time Parameter Changes**: Modify camera settings on-the-fly

## Usage

### Basic Initialization

```python
from erics_cameras import LibCameraCam

# Basic camera with auto exposure
camera = LibCameraCam(
    log_dir="./logs",
    resolution=LibCameraCam.ResolutionOption.R720P,
    camera_name="/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36"
)
```

### Manual Exposure Control

```python
# Manual exposure with specific settings
camera = LibCameraCam(
    exposure_mode=LibCameraCam.ExposureMode.MANUAL,
    exposure_time_us=5000,  # 5ms exposure
    gain_mode=LibCameraCam.GainMode.MANUAL,
    analogue_gain=2.5,      # 2.5x gain
    digital_gain=1.0
)
```

### Auto Exposure with Adjustments

```python
# Auto exposure with EV adjustment and image quality tweaks
camera = LibCameraCam(
    exposure_mode=LibCameraCam.ExposureMode.AUTO,
    exposure_value=-1.0,    # 1/2x exposure
    brightness=0.2,         # Slightly brighter
    contrast=1.2,           # Increased contrast
    saturation=1.1,         # Slightly more saturated
    sharpness=0.5           # Moderate sharpening
)
```

## Command Line Usage

### In calibrate.py

```bash
# Manual exposure control for calibration
python calibrate.py --exposure_mode manual --exposure_time 5000 --gain_mode manual --analogue_gain 2.5

# Auto exposure with EV adjustment
python calibrate.py --exposure_mode auto --exposure_value -1.0 --brightness 0.2 --contrast 1.2

# High-speed capture with short exposure
python calibrate.py --exposure_mode manual --exposure_time 1000 --gain_mode manual --analogue_gain 4.0
```

### Test Script

```bash
# Test the camera functionality
python test_libcamera.py --exposure_mode manual --exposure_time 5000 --gain_mode manual --analogue_gain 2.0

# Test auto exposure with adjustments
python test_libcamera.py --exposure_mode auto --exposure_value -0.5 --brightness 0.1
```

## Available Parameters

### Exposure Controls
- `exposure_mode`: "auto" or "manual"
- `exposure_time_us`: Exposure time in microseconds (manual mode only)
- `exposure_value`: EV adjustment (-2.0 to +2.0, log2 scale)
- `ae_enable`: Enable/disable auto-exposure

### Gain Controls
- `gain_mode`: "auto" or "manual"
- `analogue_gain`: Analogue gain multiplier (≥1.0, manual mode only)
- `digital_gain`: Digital gain multiplier

### Image Quality
- `brightness`: Brightness adjustment (-1.0 to 1.0)
- `contrast`: Contrast adjustment
- `saturation`: Saturation adjustment
- `sharpness`: Sharpness adjustment

### Advanced Controls
- `awb_enable`: Auto white balance enable/disable
- `awb_mode`: White balance mode (auto, incandescent, tungsten, fluorescent, indoor, daylight, cloudy, custom)
- `af_mode`: Autofocus mode (manual, auto, continuous)
- `lens_position`: Manual lens position in diopters

## Resolution Options

```python
# Predefined resolutions
LibCameraCam.ResolutionOption.R1080P  # 1920x1080
LibCameraCam.ResolutionOption.R720P   # 1280x720
LibCameraCam.ResolutionOption.R480P   # 640x480

# Custom resolution
camera.set_custom_resolution(1600, 900)
```

## Real-time Parameter Changes

```python
# Change exposure mode on-the-fly
camera.set_exposure_mode(LibCameraCam.ExposureMode.MANUAL)
camera.set_exposure_time(2000)  # 2ms

# Adjust image quality
camera.set_brightness(0.3)
camera.set_contrast(1.3)

# Change gain settings
camera.set_gain_mode(LibCameraCam.GainMode.MANUAL)
camera.set_analogue_gain(3.0)
```

## GStreamer Pipeline

The class automatically builds GStreamer pipelines with the appropriate libcamerasrc properties. For example:

```bash
libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36 \
    exposure-time-mode=1 exposure-time=5000 \
    analogue-gain-mode=1 analogue-gain=2.5 \
    brightness=0.2 contrast=1.2 \
    ! video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 \
    ! videoconvert ! appsink drop=1 max-buffers=1
```

## Error Handling

The class includes robust error handling for:
- Camera initialization failures
- Pipeline recreation errors
- Invalid parameter values
- GStreamer backend issues

## Performance Notes

- **Pipeline Recreation**: Changing parameters recreates the GStreamer pipeline, which may cause a brief pause
- **Manual Mode**: Manual exposure/gain modes provide consistent settings for calibration
- **Auto Mode**: Auto modes adapt to lighting changes but may vary between frames
- **Resolution**: Higher resolutions increase processing time and memory usage

## Troubleshooting

### Common Issues

1. **Camera Not Found**: Verify the `camera_name` path matches your system
2. **Permission Errors**: Ensure user has access to camera devices
3. **GStreamer Errors**: Check that GStreamer and libcamera are properly installed
4. **High Latency**: Reduce resolution or framerate for better performance

### Debug Mode

Enable debug output by setting environment variables:
```bash
export GST_DEBUG=3
export GST_DEBUG_FILE=/tmp/gstreamer.log
```

## Examples

### Calibration with Consistent Lighting

```python
# Manual mode for consistent exposure during calibration
camera = LibCameraCam(
    exposure_mode=LibCameraCam.ExposureMode.MANUAL,
    exposure_time_us=8000,      # 8ms - good for indoor lighting
    gain_mode=LibCameraCam.GainMode.MANUAL,
    analogue_gain=2.0,          # Moderate gain
    brightness=0.0,             # No brightness adjustment
    contrast=1.0,               # Normal contrast
    saturation=1.0              # Normal saturation
)
```

### High-Speed Capture

```python
# Short exposure for fast-moving objects
camera = LibCameraCam(
    exposure_mode=LibCameraCam.ExposureMode.MANUAL,
    exposure_time_us=1000,      # 1ms - very fast
    gain_mode=LibCameraCam.GainMode.MANUAL,
    analogue_gain=4.0,          # High gain to compensate
    digital_gain=1.5            # Additional digital gain
)
```

### Low-Light Conditions

```python
# Optimized for low-light environments
camera = LibCameraCam(
    exposure_mode=LibCameraCam.ExposureMode.MANUAL,
    exposure_time_us=20000,     # 20ms - longer exposure
    gain_mode=LibCameraCam.GainMode.MANUAL,
    analogue_gain=3.0,          # High analogue gain
    brightness=0.1,             # Slight brightness boost
    contrast=1.1                # Enhanced contrast
)
```
