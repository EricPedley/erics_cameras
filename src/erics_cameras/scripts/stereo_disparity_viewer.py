#!/usr/bin/env python3
"""
Stereo Disparity Viewer Script

This script displays left and right camera feeds from AVI files and computes
stereo disparity with interactive controls for video scrubbing and stereo parameters.
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Hard-coded AVI file paths - modify these to point to your stereo video files
LEFT_AVI_PATH = "/home/eric/Downloads/stereo_vid_2/left_video.avi"
RIGHT_AVI_PATH = "/home/eric/Downloads/stereo_vid_2/right_video.avi"

class StereoDisparityViewer:
    def __init__(self, left_video_path, right_video_path):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        
        # Initialize video captures
        self.left_cap = cv2.VideoCapture(left_video_path)
        self.right_cap = cv2.VideoCapture(right_video_path)
        
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            raise ValueError("Could not open one or both video files")
        
        # Get video properties
        self.total_frames = int(min(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT),
                                  self.right_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.left_cap.get(cv2.CAP_PROP_FPS)
        
        # Current frame position
        self.current_frame = 0
        
        # StereoBM parameters
        self.num_disparities = 16
        self.block_size = 15
        self.min_disparity = 0
        self.texture_threshold = 10
        self.uniqueness_ratio = 15
        self.speckle_window_size = 100
        self.speckle_range = 32
        self.disp_12_max_diff = 1
        
        # Create stereo matcher
        self.stereo = None
        self.update_stereo_matcher()
        
        # Create windows and trackbars
        self.setup_ui()
        
    def update_stereo_matcher(self):
        """Update the stereo matcher with current parameters"""
        self.stereo = cv2.StereoBM_create(
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            minDisparity=self.min_disparity,
            textureThreshold=self.texture_threshold,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            disp12MaxDiff=self.disp_12_max_diff
        )
    
    def setup_ui(self):
        """Setup OpenCV windows and trackbars"""
        # Create windows
        cv2.namedWindow("Left+Right Cameras", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Stereo Disparity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        
        # Resize windows
        cv2.resizeWindow("Left+Right Cameras", (1280, 480))
        cv2.resizeWindow("Stereo Disparity", (640, 480))
        cv2.resizeWindow("Controls", (400, 600))
        
        # Create trackbars for video control
        cv2.createTrackbar("Frame", "Controls", 0, self.total_frames - 1, self.on_frame_change)
        
        # Create trackbars for stereo parameters
        cv2.createTrackbar("Num Disparities", "Controls", self.num_disparities, 256, self.on_num_disparities_change)
        cv2.createTrackbar("Block Size", "Controls", self.block_size, 51, self.on_block_size_change)
        cv2.createTrackbar("Min Disparity", "Controls", self.min_disparity, 16, self.on_min_disparity_change)
        cv2.createTrackbar("Texture Threshold", "Controls", self.texture_threshold, 100, self.on_texture_threshold_change)
        cv2.createTrackbar("Uniqueness Ratio", "Controls", self.uniqueness_ratio, 100, self.on_uniqueness_ratio_change)
        cv2.createTrackbar("Speckle Window", "Controls", self.speckle_window_size, 200, self.on_speckle_window_change)
        cv2.createTrackbar("Speckle Range", "Controls", self.speckle_range, 100, self.on_speckle_range_change)
        cv2.createTrackbar("Disp12 Max Diff", "Controls", self.disp_12_max_diff, 100, self.on_disp12_max_diff_change)
        
        # Set initial frame position
        cv2.setTrackbarPos("Frame", "Controls", 0)
    
    def on_frame_change(self, value):
        """Callback for frame trackbar changes"""
        self.current_frame = value
        self.seek_to_frame(value)
    
    def on_num_disparities_change(self, value):
        """Callback for num disparities trackbar changes"""
        # Ensure value is divisible by 16
        self.num_disparities = (value // 16) * 16
        if self.num_disparities == 0:
            self.num_disparities = 16
        cv2.setTrackbarPos("Num Disparities", "Controls", self.num_disparities)
        self.update_stereo_matcher()
    
    def on_block_size_change(self, value):
        """Callback for block size trackbar changes"""
        # Ensure value is odd
        self.block_size = value if value % 2 == 1 else value + 1
        cv2.setTrackbarPos("Block Size", "Controls", self.block_size)
        self.update_stereo_matcher()
    
    def on_min_disparity_change(self, value):
        """Callback for min disparity trackbar changes"""
        self.min_disparity = value
        self.update_stereo_matcher()
    
    def on_texture_threshold_change(self, value):
        """Callback for texture threshold trackbar changes"""
        self.texture_threshold = value
        self.update_stereo_matcher()
    
    def on_uniqueness_ratio_change(self, value):
        """Callback for uniqueness ratio trackbar changes"""
        self.uniqueness_ratio = value
        self.update_stereo_matcher()
    
    def on_speckle_window_change(self, value):
        """Callback for speckle window trackbar changes"""
        self.speckle_window_size = value
        self.update_stereo_matcher()
    
    def on_speckle_range_change(self, value):
        """Callback for speckle range trackbar changes"""
        self.speckle_range = value
        self.update_stereo_matcher()
    
    def on_disp12_max_diff_change(self, value):
        """Callback for disp12 max diff trackbar changes"""
        self.disp_12_max_diff = value
        self.update_stereo_matcher()
    
    def seek_to_frame(self, frame_number):
        """Seek both video captures to the specified frame"""
        self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def read_frame_pair(self):
        """Read a frame from both video captures"""
        ret1, left_frame = self.left_cap.read()
        ret2, right_frame = self.right_cap.read()
        
        if not ret1 or not ret2:
            return None, None
        
        return left_frame, right_frame
    
    def compute_disparity(self, left_frame, right_frame):
        """Compute stereo disparity from left and right frames"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Normalize disparity for display
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for better visualization
        disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        return disparity_colored
    
    def add_info_text(self, frame, text_lines):
        """Add information text overlay to a frame"""
        y_offset = 30
        for line in text_lines:
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 25
    
    def create_concatenated_view(self, left_frame, right_frame):
        """Create a concatenated view of left and right frames with horizontal lines"""
        # Ensure both frames have the same dimensions
        height, width = left_frame.shape[:2]
        
        # Concatenate horizontally
        concatenated = np.hstack([left_frame, right_frame])
        
        # Add horizontal lines every 50 pixels
        for y in range(0, height, 50):
            cv2.line(concatenated, (0, y), (concatenated.shape[1], y), (0, 255, 0), 1)
        
        # Add a vertical separator line between left and right images
        cv2.line(concatenated, (width, 0), (width, height), (255, 0, 0), 2)
        
        return concatenated
    
    def run(self):
        """Main loop for the stereo disparity viewer"""
        print(f"Starting stereo disparity viewer")
        print(f"Left video: {self.left_video_path}")
        print(f"Right video: {self.right_video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")
        print("Press 'q' to quit, 's' to save current disparity image")
        
        while True:
            # Read current frame pair
            left_frame, right_frame = self.read_frame_pair()
            
            if left_frame is None or right_frame is None:
                print("End of video reached")
                break
            
            # Compute disparity
            disparity_frame = self.compute_disparity(left_frame, right_frame)
            
            # Create concatenated view
            concatenated_frame = self.create_concatenated_view(left_frame, right_frame)
            
            # Add info overlays
            frame_info = [
                f"Frame: {self.current_frame}/{self.total_frames}",
                f"Time: {self.current_frame/self.fps:.2f}s",
                f"Num Disparities: {self.num_disparities}",
                f"Block Size: {self.block_size}"
            ]
            
            self.add_info_text(concatenated_frame, frame_info)
            self.add_info_text(disparity_frame, frame_info)
            
            # Display frames
            cv2.imshow("Left+Right Cameras", concatenated_frame)
            cv2.imshow("Stereo Disparity", disparity_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current disparity image
                filename = f"disparity_frame_{self.current_frame:04d}.png"
                cv2.imwrite(filename, disparity_frame)
                print(f"Saved disparity image: {filename}")
            elif key == ord(' '):
                # Spacebar to pause/unpause
                cv2.waitKey(0)
        
        # Cleanup
        self.left_cap.release()
        self.right_cap.release()
        cv2.destroyAllWindows()
        print("Stereo disparity viewer closed")

def main():
    """Main function"""
    # Check if video files exist
    if not os.path.exists(LEFT_AVI_PATH):
        print(f"Error: Left video file not found: {LEFT_AVI_PATH}")
        print("Please update the LEFT_AVI_PATH variable in the script")
        return
    
    if not os.path.exists(RIGHT_AVI_PATH):
        print(f"Error: Right video file not found: {RIGHT_AVI_PATH}")
        print("Please update the RIGHT_AVI_PATH variable in the script")
        return
    
    try:
        viewer = StereoDisparityViewer(LEFT_AVI_PATH, RIGHT_AVI_PATH)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
