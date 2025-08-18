from .camera import Camera, ImageMetadata
from .gst_cam import GstCamera
import time
from pathlib import Path
from .camera_types import Image
import cv2 as cv
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation


class LibCameraCam(Camera):
    class ResolutionOption(Enum):
        R1080P = (1920, 1080)
        R720P = (1280, 720)
        R480P = (640, 480)
        R_CUSTOM = None  # For custom resolution

    class ExposureMode(Enum):
        AUTO = "auto"
        MANUAL = "manual"

    class GainMode(Enum):
        AUTO = "auto"
        MANUAL = "manual"

    def __init__(
        self,
        log_dir: str | Path | None = None,
        resolution: ResolutionOption = ResolutionOption.R720P,
        camera_name: str = "/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36",
        framerate: int = 30,
        exposure_mode: ExposureMode = ExposureMode.AUTO,
        exposure_time_us: int = 10000,  # 10ms default
        gain_mode: GainMode = GainMode.AUTO,
        analogue_gain: float = 2.0,
        digital_gain: float = 1.0,
        ae_enable: bool = True,
        exposure_value: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        sharpness: float = 0.0,
        awb_enable: bool = True,
        awb_mode: str = "auto",
        af_mode: str = "manual",
        lens_position: float = 0.0,
    ):
        super().__init__(log_dir)
        
        self._resolution = resolution
        self._camera_name = camera_name
        self._framerate = framerate
        self._exposure_mode = exposure_mode
        self._exposure_time_us = exposure_time_us
        self._gain_mode = gain_mode
        self._analogue_gain = analogue_gain
        self._digital_gain = digital_gain
        self._ae_enable = ae_enable
        self._exposure_value = exposure_value
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation
        self._sharpness = sharpness
        self._awb_enable = awb_enable
        self._awb_mode = awb_mode
        self._af_mode = af_mode
        self._lens_position = lens_position
        
        # Build the GStreamer pipeline
        pipeline = self._build_pipeline()
        print(f"LibCamera pipeline: {pipeline}")
        
        self._cam = GstCamera(log_dir, pipeline)

    def _build_pipeline(self) -> str:
        """Build the GStreamer pipeline with current settings."""
        # Base pipeline
        pipeline_parts = [
            f"libcamerasrc camera-name={self._camera_name}"
        ]
        
        # Add exposure controls
        if self._exposure_mode == self.ExposureMode.MANUAL:
            pipeline_parts.append(f"exposure-time-mode=1 exposure-time={self._exposure_time_us}")
        else:
            pipeline_parts.append("exposure-time-mode=0")
            
        # Add gain controls
        if self._gain_mode == self.GainMode.MANUAL:
            pipeline_parts.append(f"analogue-gain-mode=1 analogue-gain={self._analogue_gain}")
        else:
            pipeline_parts.append("analogue-gain-mode=0")
            
        # Add auto-exposure controls
        pipeline_parts.append(f"ae-enable={str(self._ae_enable).lower()}")
        if self._exposure_value != 0.0:
            pipeline_parts.append(f"exposure-value={self._exposure_value}")
            
        # Add image quality controls
        if self._brightness != 0.0:
            pipeline_parts.append(f"brightness={self._brightness}")
        if self._contrast != 0.0:
            pipeline_parts.append(f"contrast={self._contrast}")
        if self._saturation != 0.0:
            pipeline_parts.append(f"saturation={self._saturation}")
        if self._sharpness != 0.0:
            pipeline_parts.append(f"sharpness={self._sharpness}")
            
        # Add AWB controls
        pipeline_parts.append(f"awb-enable={str(self._awb_enable).lower()}")
        if self._awb_mode != "auto":
            pipeline_parts.append(f"awb-mode={self._awb_mode}")
            
        # Add AF controls
        pipeline_parts.append(f"af-mode={self._af_mode}")
        if self._lens_position != 0.0:
            pipeline_parts.append(f"lens-position={self._lens_position}")
            
        # Add digital gain
        if self._digital_gain != 1.0:
            pipeline_parts.append(f"digital-gain={self._digital_gain}")
            
        # Add video format and sink
        if self._resolution == self.ResolutionOption.R_CUSTOM:
            # Use custom resolution if set
            width, height = self._custom_resolution
        else:
            width, height = self._resolution.value
            
        pipeline_parts.extend([
            "!",
            f"video/x-raw,format=BGR,width={width},height={height},framerate={self._framerate}/1",
            "!",
            "videoconvert",
            "!",
            "appsink drop=1 max-buffers=1"
        ])
        
        return " ".join(pipeline_parts)

    def set_exposure_mode(self, mode: ExposureMode):
        """Set exposure mode (auto/manual)."""
        self._exposure_mode = mode
        self._recreate_pipeline()

    def set_exposure_time(self, time_us: int):
        """Set exposure time in microseconds (only effective in manual mode)."""
        self._exposure_time_us = time_us
        if self._exposure_mode == self.ExposureMode.MANUAL:
            self._recreate_pipeline()

    def set_gain_mode(self, mode: GainMode):
        """Set gain mode (auto/manual)."""
        self._gain_mode = mode
        self._recreate_pipeline()

    def set_analogue_gain(self, gain: float):
        """Set analogue gain (only effective in manual mode)."""
        self._analogue_gain = gain
        if self._gain_mode == self.GainMode.MANUAL:
            self._recreate_pipeline()

    def set_digital_gain(self, gain: float):
        """Set digital gain."""
        self._digital_gain = gain
        self._recreate_pipeline()

    def set_ae_enable(self, enable: bool):
        """Enable/disable auto-exposure."""
        self._ae_enable = enable
        self._recreate_pipeline()

    def set_exposure_value(self, ev: float):
        """Set exposure value (EV) adjustment."""
        self._exposure_value = ev
        self._recreate_pipeline()

    def set_brightness(self, brightness: float):
        """Set brightness adjustment (-1.0 to 1.0)."""
        self._brightness = brightness
        self._recreate_pipeline()

    def set_contrast(self, contrast: float):
        """Set contrast adjustment."""
        self._contrast = contrast
        self._recreate_pipeline()

    def set_saturation(self, saturation: float):
        """Set saturation adjustment."""
        self._saturation = saturation
        self._recreate_pipeline()

    def set_sharpness(self, sharpness: float):
        """Set sharpness adjustment."""
        self._sharpness = sharpness
        self._recreate_pipeline()

    def set_awb_enable(self, enable: bool):
        """Enable/disable auto white balance."""
        self._awb_enable = enable
        self._recreate_pipeline()

    def set_awb_mode(self, mode: str):
        """Set AWB mode (auto, incandescent, tungsten, fluorescent, indoor, daylight, cloudy, custom)."""
        self._awb_mode = mode
        self._recreate_pipeline()

    def set_af_mode(self, mode: str):
        """Set autofocus mode (manual, auto, continuous)."""
        self._af_mode = mode
        self._recreate_pipeline()

    def set_lens_position(self, position: float):
        """Set lens position (diopters)."""
        self._lens_position = position
        self._recreate_pipeline()

    def set_custom_resolution(self, width: int, height: int):
        """Set custom resolution."""
        self._custom_resolution = (width, height)
        self._recreate_pipeline()

    def _recreate_pipeline(self):
        """Recreate the camera with new pipeline settings."""
        # Close existing camera
        if hasattr(self._cam, 'cap') and self._cam.cap is not None:
            self._cam.cap.release()
        
        # Create new pipeline and camera
        pipeline = self._build_pipeline()
        print(f"Recreating pipeline: {pipeline}")
        self._cam = GstCamera(self.log_dir, pipeline)

    def take_image(self) -> Image[np.ndarray] | None:
        """Take an image from the camera."""
        frame = self._cam.getFrame()
        if frame is None:
            return None
        return Image(frame)

    def get_focal_length_px(self):
        """Get focal length in pixels (approximate)."""
        if self._resolution == self.ResolutionOption.R_CUSTOM:
            width = self._custom_resolution[0]
        else:
            width = self._resolution.value[0]
        # Approximate focal length based on typical Raspberry Pi camera
        return 1154 * width / 1920

    def get_current_settings(self) -> dict:
        """Get current camera settings as a dictionary."""
        return {
            'exposure_mode': self._exposure_mode.value,
            'exposure_time_us': self._exposure_time_us,
            'gain_mode': self._gain_mode.value,
            'analogue_gain': self._analogue_gain,
            'digital_gain': self._digital_gain,
            'ae_enable': self._ae_enable,
            'exposure_value': self._exposure_value,
            'brightness': self._brightness,
            'contrast': self._contrast,
            'saturation': self._saturation,
            'sharpness': self._sharpness,
            'awb_enable': self._awb_enable,
            'awb_mode': self._awb_mode,
            'af_mode': self._af_mode,
            'lens_position': self._lens_position,
            'resolution': self._resolution.value if self._resolution != self.ResolutionOption.R_CUSTOM else self._custom_resolution,
            'framerate': self._framerate,
        }
