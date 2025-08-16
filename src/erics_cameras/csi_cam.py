from .gst_cam import GstCamera
from .camera import Camera
from enum import Enum
from pathlib import Path
import cv2 as cv
from .camera_types import Image


class CSICam(Camera):
    class ResolutionOption(Enum):
        R4K = (3840, 2160)
        R1080P = (1920, 1080)
        R720P = (1280, 720)
        R480P = (640, 480)

    def __init__(
        self,
        log_dir: str | Path | None = None,
        resolution: ResolutionOption = ResolutionOption.R4K,
        flipped=True,  # because of how they're mounted we might have to flip them sometimes.
    ):
        super().__init__(log_dir)
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                resolution.value[0],
                resolution.value[1],
                30,
                0 if flipped else 2,
                resolution.value[0],
                resolution.value[1],
            )
        )

        self._camera = GstCamera(pipeline)
        self._resolution = resolution
        self._flipped = flipped

    def take_image(self) -> Image:
        frame = self._camera.getFrame()
        if frame is None:
            return None
        if self._flipped:
            frame = cv.rotate(frame, cv.ROTATE_180)
        return Image(frame)