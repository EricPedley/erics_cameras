from numpy import ndarray
from .camera_types import Image
from .camera import Camera 
import numpy as np


class MockCamera(Camera):
    '''A camera that returns random noise images'''
    def __init__(self, log_dir, resolution=(1920, 1080)):
        super().__init__(None)
        self.resolution = resolution

    def take_image(self) -> Image[ndarray] | None:
        return Image(np.random.randint(0, 255, size=(self.resolutionp[1], self.resolution[0], 3)))

    def get_focal_length_px(self) -> float:
        return 1