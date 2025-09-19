from erics_cameras.camera import Camera
from erics_cameras.camera_types import Image
from erics_cameras.usb_cam import USBCam
from pathlib import Path
import cv2

class StereoCam:
    '''
    Class representing a stereo camera setup with left and right cameras.
    Options are:
    1. provide two Camera objects as the left and right
    2. use a single USB cam that provides both streams in a single wide frame
       (in this case, the frame is split in half to get left and right images)
    '''
    def __init__(self, left_cam: Camera | None = None, right_cam: Camera | None = None, usb_cam: USBCam | None = None):
        if usb_cam is not None:
            if left_cam is not None or right_cam is not None:
                raise ValueError("If usb_cam is provided, left_cam and right_cam must be None")
            self._usb_cam = usb_cam
            self._left_cam = None
            self._right_cam = None
        elif left_cam is not None and right_cam is not None:
            self._left_cam = left_cam
            self._right_cam = right_cam
            self._usb_cam = None
        else:
            raise ValueError("Either both left_cam and right_cam must be provided, or usb_cam must be provided")
    
    def get_left_camera(self) -> Camera:
        '''
        Gets the left camera. Constructs a special Camera object if using a single USB cam.
        '''
        if self._usb_cam is not None:
            return _StereoUSBCamWrapper(self._usb_cam, side='left')
        elif self._left_cam is not None:
            return self._left_cam
        else:
            raise RuntimeError("StereoCam is not properly initialized")
    
    def get_right_camera(self) -> Camera:
        '''
        Gets the right camera. Constructs a special Camera object if using a single USB cam.
        '''
        if self._usb_cam is not None:
            return _StereoUSBCamWrapper(self._usb_cam, side='right')
        elif self._right_cam is not None:
            return self._right_cam
        else:
            raise RuntimeError("StereoCam is not properly initialized")
    
    def get_image_pair(self) -> tuple[Image, Image]:
        '''
        Gets a synchronized pair of images from the left and right cameras.
        '''
        if self._usb_cam is None:
            left_image = self.get_left_camera().take_image()
            right_image = self.get_right_camera().take_image()
            return left_image, right_image
        else:
            combined_image = self._usb_cam.take_image().get_array()
            height, width = combined_image.shape[:2]
            left_image = combined_image[:, :width//2]
            right_image = combined_image[:, width//2:]
            return Image(left_image), Image(right_image)

class _StereoUSBCamWrapper(Camera):
    '''
    A wrapper around a single USBCam to provide left or right images by splitting the frame.
    '''
    def __init__(self, usb_cam: USBCam, side: str):
        super().__init__(usb_cam._log_dir)
        if side not in ['left', 'right']:
            raise ValueError("side must be 'left' or 'right'")
        self._usb_cam = usb_cam
        self._side = side
    
    def take_image(self):
        frame = self._usb_cam.take_image()
        height, width = frame.shape[:2]
        if self._side == 'left':
            return frame[:, :width//2]
        else:
            return frame[:, width//2:]
    
class StereoReplayCam:
    def __init__(self, folder_path: str | Path):
        self.folder_path = Path(folder_path)
        self.index = 0
        
        # Count available image pairs to know when to stop
        left_images = sorted(list(self.folder_path.glob("left_*.png")))
        right_images = sorted(list(self.folder_path.glob("right_*.png")))
        self.total_pairs = min(len(left_images), len(right_images))
    
    def take_image(self):
        if self.index >= self.total_pairs:
            raise RuntimeError("No more image pairs available")
            
        left_path = self.folder_path / f"left_{self.index}.png"
        right_path = self.folder_path / f"right_{self.index}.png"
        
        if not left_path.exists() or not right_path.exists():
            raise RuntimeError(f"Missing image files: {left_path} or {right_path}")
            
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))
        
        if left_image is None or right_image is None:
            raise RuntimeError(f"Failed to load image files: {left_path} or {right_path}")
        
        self.index += 1
        return left_image, right_image
    
    def get_total_frames(self):
        return self.total_pairs
