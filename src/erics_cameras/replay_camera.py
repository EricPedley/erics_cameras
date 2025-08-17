import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
import warnings

from .camera import Camera
from .camera_types import Image, HWC


class ReplayCamera(Camera):
    """
    A camera implementation that replays images from a folder or video file.
    
    This camera can read images sequentially from a folder or extract frames from a video file.
    When all images/frames have been read, it raises an error.
    """
    
    def __init__(self, source_path: str | Path, log_dir: str | Path | None = None):
        """
        Initialize the ReplayCamera.
        
        Parameters
        ----------
        source_path : str | Path
            Path to either a folder containing images or a video file
        log_dir : str | Path | None, optional
            Directory for logging images and metadata, by default None
        """
        super().__init__(log_dir)
        
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Determine if source is a folder or video file
        if self.source_path.is_dir():
            self._init_from_folder()
        elif self._is_video_file(self.source_path):
            self._init_from_video()
        else:
            raise ValueError(f"Source path must be a directory or video file: {source_path}")
        
        # Current frame index
        self.current_index = 0
        
        print(f"ReplayCamera initialized with {self.total_frames} frames from {source_path}")
    
    def _is_video_file(self, path: Path) -> bool:
        """Check if the given path is a video file."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return path.suffix.lower() in video_extensions
    
    def _init_from_folder(self):
        """Initialize camera from an image folder."""
        # Get list of image files in the folder
        self.image_files = self._get_image_files()
        if not self.image_files:
            raise ValueError(f"No image files found in folder: {self.source_path}")
        
        self.total_frames = len(self.image_files)
        self.source_type = "folder"
    
    def _init_from_video(self):
        """Initialize camera from a video file."""
        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.source_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            raise ValueError(f"Invalid frame count in video: {self.total_frames}")
        
        self.source_type = "video"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
        for file_path in self.source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # Sort files alphabetically for consistent ordering
        return sorted(image_files)
    
    def take_image(self) -> Image[np.ndarray] | None:
        """
        Take an image by reading the next frame from the source.
        
        Returns
        -------
        Image[np.ndarray]
            The next frame from the source
            
        Raises
        ------
        RuntimeError
            When all frames from the source have been read
        """
        if self.current_index >= self.total_frames:
            raise RuntimeError(
                f"All {self.total_frames} frames from source have been exhausted. "
                f"Reset the camera or provide more frames."
            )
        
        try:
            if self.source_type == "folder":
                frame = self._take_image_from_folder()
            else:  # video
                frame = self._take_frame_from_video()
            
            if frame is None:
                warnings.warn(f"Failed to read frame at index {self.current_index}, {self.image_files[self.current_index]}")
                # Skip this frame and try the next one
                self.current_index += 1
                return self.take_image()  # Recursively try the next frame
            
            # Create Image object and increment counter
            image = Image(frame, HWC)
            self.current_index += 1
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Error reading frame at index {self.current_index}: {str(e)}")
    
    def _take_image_from_folder(self) -> np.ndarray | None:
        """Read the next image from the folder."""
        image_path = self.image_files[self.current_index]
        frame = cv2.imread(str(image_path))
        return frame
    
    def _take_frame_from_video(self) -> np.ndarray | None:
        """Read the next frame from the video."""
        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def reset(self):
        """
        Reset the camera to start reading from the first frame again.
        """
        self.current_index = 0
        if self.source_type == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"ReplayCamera reset to first frame")
    
    def get_remaining_frames(self) -> int:
        """
        Get the number of remaining frames to be read.
        
        Returns
        -------
        int
            Number of remaining frames
        """
        return max(0, self.total_frames - self.current_index)
    
    def get_total_frames(self) -> int:
        """
        Get the total number of frames from the source.
        
        Returns
        -------
        int
            Total number of frames
        """
        return self.total_frames
    
    def get_current_source_info(self) -> str | None:
        """
        Get information about the current frame being read.
        
        Returns
        -------
        str | None
            Information about current frame, or None if all frames exhausted
        """
        if self.current_index < self.total_frames:
            if self.source_type == "folder":
                return f"Image: {self.image_files[self.current_index].name}"
            else:  # video
                return f"Frame: {self.current_index}/{self.total_frames} ({(self.current_index/self.total_frames)*100:.1f}%)"
        return None
    
    def get_video_info(self) -> dict | None:
        """
        Get video information if source is a video file.
        
        Returns
        -------
        dict | None
            Video information dict, or None if source is not a video
        """
        if self.source_type == "video":
            return {
                "fps": self.fps,
                "width": self.frame_width,
                "height": self.frame_height,
                "total_frames": self.total_frames,
                "duration_seconds": self.total_frames / self.fps if self.fps > 0 else None
            }
        return None

    def __del__(self):
        """Clean up resources when the camera is destroyed."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def close(self):
        """Explicitly close the camera and release resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
