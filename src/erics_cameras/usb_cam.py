from .camera import Camera, ImageMetadata
from .gst_cam import GstCamera
import time
from pathlib import Path
from .camera_types import Image
import cv2 as cv
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation

# GStreamer Python bindings for property control
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False


class USBCam(Camera):
    class ResolutionOption(Enum):
        R1080P = (1920, 1080)
        R720P = (1280, 720)
        R480P = (640, 480)

    def __init__(
        self,
        log_dir: str | Path | None = None,
        resolution: ResolutionOption = ResolutionOption.R720P,
        flipped=False,  # because of how they're mounted we might have to flip them sometimes.
        video_path="/dev/video0",
    ):
        super().__init__(log_dir)
        self._log_dir = log_dir
        self._flipped = (
            flipped  # TODO: just put this in the gstreamer pipeline if we need speed
        )
        # laptop params:
        self._pipeline_str = (
            rf"v4l2src device={video_path} io-mode=2 ! "
            rf"image/jpeg,width={resolution.value[0]},height={resolution.value[1]},framerate=30/1 ! "
            r"jpegdec ! "
            r"videoconvert ! "
            r"video/x-raw, format=BGR ! "
            r"appsink drop=true max-buffers=1"
        )
        print(self._pipeline_str)
        self._resolution = resolution
        self._video_path = video_path
        self._cam = GstCamera(log_dir, self._pipeline_str)
        
        # Store current property values
        self._brightness = 0
        self._contrast = 0
        self._saturation = 0
        self._hue = 0

    def _create_pipeline_with_properties(self, brightness=None, contrast=None, saturation=None, hue=None):
        """Create a new pipeline string with the specified properties"""
        # Use current values if not specified
        if brightness is not None:
            self._brightness = brightness
        if contrast is not None:
            self._contrast = contrast
        if saturation is not None:
            self._saturation = saturation
        if hue is not None:
            self._hue = hue
            
        # Build property string
        properties = []
        if self._brightness != 0:
            properties.append(f"brightness={self._brightness}")
        if self._contrast != 0:
            properties.append(f"contrast={self._contrast}")
        if self._saturation != 0:
            properties.append(f"saturation={self._saturation}")
        if self._hue != 0:
            properties.append(f"hue={self._hue}")
            
        prop_str = " ".join(properties)
        
        # Create new pipeline
        new_pipeline = (
            rf"v4l2src device={self._video_path} io-mode=2 {prop_str} ! "
            rf"image/jpeg,width={self._resolution.value[0]},height={self._resolution.value[1]},framerate=30/1 ! "
            r"jpegdec ! "
            r"videoconvert ! "
            r"video/x-raw, format=BGR ! "
            r"appsink drop=true max-buffers=1"
        )
        return new_pipeline

    def update_properties(self, brightness=None, contrast=None, saturation=None, hue=None):
        """Update camera properties by recreating the pipeline"""
        if not GST_AVAILABLE:
            print("Warning: GStreamer Python bindings not available. Cannot update properties.")
            return False
            
        try:
            # Close current camera
            if hasattr(self._cam, 'close'):
                self._cam.close()
            
            # Create new pipeline with updated properties
            new_pipeline = self._create_pipeline_with_properties(brightness, contrast, saturation, hue)
            
            # Create new camera with updated pipeline
            self._cam = GstCamera(self._log_dir, new_pipeline)
            print(f"Updated pipeline: {new_pipeline}")
            return True
        except Exception as e:
            print(f"Failed to update properties: {e}")
            # Try to restore original pipeline
            try:
                self._cam = GstCamera(self._log_dir, self._pipeline_str)
            except:
                pass
            return False

    def set_brightness(self, value: int):
        """Set brightness (-2147483648 to 2147483647)"""
        return self.update_properties(brightness=value)
        
    def set_contrast(self, value: int):
        """Set contrast (-2147483648 to 2147483647)"""
        return self.update_properties(contrast=value)
        
    def set_saturation(self, value: int):
        """Set saturation (-2147483648 to 2147483647)"""
        return self.update_properties(saturation=value)
        
    def set_hue(self, value: int):
        """Set hue (-2147483648 to 2147483647)"""
        return self.update_properties(hue=value)
        
    def get_properties(self):
        """Get current property values"""
        return {
            'brightness': self._brightness,
            'contrast': self._contrast,
            'saturation': self._saturation,
            'hue': self._hue
        }

    def take_image(self) -> Image[np.ndarray] | None:
        frame = self._cam.getFrame()
        if frame is None:
            return None
        if self._flipped:
            frame = cv.rotate(frame, cv.ROTATE_180)
        return Image(frame)


    def get_focal_length_px(self):
        return 1154 * self._resolution.value[0] / 1920
        """
        [[     1154.4           0      670.62]
         [          0      1158.6      836.27]
         [          0           0           1]]
        [[   -0.29957    0.099084    0.031633   -0.019682   -0.012533           0           0           0    0.082185   -0.017815    -0.10748    0.015057]]
        """
