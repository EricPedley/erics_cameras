from .camera import Camera, ImageMetadata
from .gst_cam import GstCamera
import time
from pathlib import Path
from .camera_types import Image
import cv2 as cv
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation


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
        self._flipped = (
            flipped  # TODO: just put this in the gstreamer pipeline if we need speed
        )
        # laptop params:
        pipeline = (
            rf"v4l2src device={video_path} io-mode=2 ! "
            rf"image/jpeg,width={resolution.value[0]},height={resolution.value[1]},framerate=30/1 ! "
            r"jpegdec ! "
            r"videoconvert ! "
            r"video/x-raw, format=BGR ! "
            r"appsink drop=true max-buffers=1"
        )
        print(pipeline)
        self._resolution = resolution
        self._cam = GstCamera(log_dir, pipeline)

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
