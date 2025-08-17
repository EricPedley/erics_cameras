import os
from pathlib import Path
from typing import List
import cv2
import numpy as np

from .camera import Camera
from .camera_types import Image


class ImageFolderCamera(Camera):
    """
    A camera implementation that reads images sequentially from a folder.
    
    This camera reads images from a specified folder in alphabetical order.
    When all images have been read, it raises an error.
    """
    
    def __init__(self, image_folder: str | Path, log_dir: str | Path | None = None):
        """
        Initialize the ImageFolderCamera.
        
        Parameters
        ----------
        image_folder : str | Path
            Path to the folder containing images
        log_dir : str | Path | None, optional
            Directory for logging images and metadata, by default None
        """
        super().__init__(log_dir)
        
        self.image_folder = Path(image_folder)
        if not self.image_folder.exists():
            raise ValueError(f"Image folder does not exist: {image_folder}")
        if not self.image_folder.is_dir():
            raise ValueError(f"Path is not a directory: {image_folder}")
        
        # Get list of image files in the folder
        self.image_files = self._get_image_files()
        if not self.image_files:
            raise ValueError(f"No image files found in folder: {image_folder}")
        
        # Current image index
        self.current_index = 0
        
        print(f"ImageFolderCamera initialized with {len(self.image_files)} images from {image_folder}")
    
    def _get_image_files(self) -> List[Path]:
        """
        Get a sorted list of image files from the folder.
        
        Returns
        -------
        List[Path]
            Sorted list of image file paths
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        image_files = []
        for file_path in self.image_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # Sort files alphabetically for consistent ordering
        return sorted(image_files)
    
    def take_image(self) -> Image[np.ndarray] | None:
        """
        Take an image by reading the next image file from the folder.
        
        Returns
        -------
        Image[np.ndarray]
            The next image from the folder
            
        Raises
        ------
        RuntimeError
            When all images in the folder have been read
        """
        if self.current_index >= len(self.image_files):
            raise RuntimeError(
                f"All {len(self.image_files)} images from folder have been exhausted. "
                f"Reset the camera or provide more images."
            )
        
        # Read the current image file
        image_path = self.image_files[self.current_index]
        try:
            # Read image using OpenCV (BGR format)
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            
            # Create Image object and increment counter
            image = Image(frame)
            self.current_index += 1
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Error reading image {image_path}: {str(e)}")
    
    def reset(self):
        """
        Reset the camera to start reading from the first image again.
        """
        self.current_index = 0
        print(f"ImageFolderCamera reset to first image")
    
    def get_remaining_images(self) -> int:
        """
        Get the number of remaining images to be read.
        
        Returns
        -------
        int
            Number of remaining images
        """
        return max(0, len(self.image_files) - self.current_index)
    
    def get_total_images(self) -> int:
        """
        Get the total number of images in the folder.
        
        Returns
        -------
        int
            Total number of images
        """
        return len(self.image_files)
    
    def get_current_image_path(self) -> Path | None:
        """
        Get the path of the current image being read.
        
        Returns
        -------
        Path | None
            Path to current image, or None if all images exhausted
        """
        if self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None
