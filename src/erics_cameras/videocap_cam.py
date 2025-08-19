from .camera import Camera, ImageMetadata
from .gst_cam import GstCamera
import time
from pathlib import Path
from .camera_types import Image
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation
import cv2

class VideoCapCam(Camera):

    def __init__(
        self,
        log_dir: str | Path | None = None,
        flipped=False,  # because of how they're mounted we might have to flip them sometimes.
        video_path="/dev/video0",
    ):
        super().__init__(log_dir)
        self._log_dir = log_dir
        self._flipped = (
            flipped  # TODO: just put this in the gstreamer pipeline if we need speed
        )
        # laptop params:
        self._video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Store current property values
        self._brightness = 0
        self._contrast = 0
        self._saturation = 0
        self._hue = 0

    def take_image(self) -> Image[np.ndarray] | None:
        ret, frame = self.cap.read()
        if frame is None:
            return None
        if self._flipped:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return Image(frame)


    def get_focal_length_px(self):
        return 1154 * self._resolution.value[0] / 1920
        """
        [[     1154.4           0      670.62]
         [          0      1158.6      836.27]
         [          0           0           1]]
        [[   -0.29957    0.099084    0.031633   -0.019682   -0.012533           0           0           0    0.082185   -0.017815    -0.10748    0.015057]]
        """

if __name__ == "__main__":
    cam = VideoCapCam(None)
    while True:
        frame = cam.take_image().get_array()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        