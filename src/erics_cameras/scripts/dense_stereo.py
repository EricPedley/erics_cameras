import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera calibration parameters from your setup
R = np.eye(3)
t = [0.116, 0, 0]

cam_mat = np.array([[297.80062345, 0., 685.72493754],
                   [0., 298.63865273, 451.61133244],
                   [0., 0., 1.]])

dist_coeffs = np.array([[-0.20148179, 0.03270111, 0., 0., -0.00211291]])

class DenseDepthEstimator:
    def __init__(self):
        # Camera parameters
        self.K1 = cam_mat
        self.K2 = cam_mat
        self.dist1 = dist_coeffs
        self.dist2 = dist_coeffs
        self.R = R
        self.T = np.array(t).reshape((3, 1))
        
        # Baseline (distance between cameras in meters)
        self.baseline = np.linalg.norm(self.T)
        
        # Image size (will be updated from actual frames)
        self.img_size = (1280, 960)
        
        # Stereo rectification maps (computed once)
        self.map1x, self.map1y, self.map2x, self.map2y = None, None, None, None
        self.Q = None  # Disparity-to-depth mapping matrix
        self.roi1, self.roi2 = None, None
        
        # Initialize stereo matchers
        self.init_stereo_matchers()
        
        # Compute rectification maps
        self.compute_rectification_maps()
    
    def init_stereo_matchers(self):
        """Initialize different stereo matching algorithms"""
        # StereoBM - Fast but less accurate
        self.stereo_bm = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        # StereoSGBM - Slower but more accurate
        window_size = 3
        min_disp = 0
        num_disp = 16*5  # Must be divisible by 16
        
        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8*3*window_size**2,
            P2=32*3*window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Try to initialize WLS Filter (requires opencv-contrib-python)
        self.has_wls = False
        try:
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_sgbm)
            self.stereo_sgbm_right = cv2.ximgproc.createRightMatcher(self.stereo_sgbm)
            
            # WLS Filter parameters
            self.wls_filter.setLambda(80000)
            self.wls_filter.setSigmaColor(1.2)
            self.has_wls = True
            print("WLS filtering available")
        except AttributeError:
            print("WLS filtering not available (opencv-contrib-python not installed)")
            self.wls_filter = None
            self.stereo_sgbm_right = None
        
        # Current matcher selection
        self.current_matcher = 'sgbm'  # 'bm', 'sgbm', 'wls' (if available)
    
    def compute_rectification_maps(self):
        """Compute stereo rectification maps"""
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.dist1,
            self.K2, self.dist2,
            self.img_size,
            self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9
        )
        
        # Compute rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.dist1, R1, P1, self.img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.dist2, R2, P2, self.img_size, cv2.CV_32FC1
        )
        
        self.Q = Q
        self.roi1 = roi1
        self.roi2 = roi2
        
        print("Rectification maps computed successfully")
        print(f"ROI1: {roi1}, ROI2: {roi2}")
    
    def rectify_images(self, img1, img2):
        """Rectify stereo image pair"""
        # Apply rectification
        img1_rect = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        return img1_rect, img2_rect
    
    def compute_disparity(self, img1_rect, img2_rect):
        """Compute disparity map using selected stereo matcher"""
        # Convert to grayscale
        if len(img1_rect.shape) == 3:
            gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1_rect, img2_rect
        
        if self.current_matcher == 'bm':
            # StereoBM
            disparity = self.stereo_bm.compute(gray1, gray2)
        elif self.current_matcher == 'sgbm':
            # StereoSGBM
            disparity = self.stereo_sgbm.compute(gray1, gray2)
        elif self.current_matcher == 'wls' and self.has_wls:
            # SGBM with WLS post-filtering
            disp_left = self.stereo_sgbm.compute(gray1, gray2)
            disp_right = self.stereo_sgbm_right.compute(gray2, gray1)
            disparity = self.wls_filter.filter(disp_left, gray1, None, disp_right)
        else:
            # Fallback to SGBM if WLS not available
            disparity = self.stereo_sgbm.compute(gray1, gray2)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def disparity_to_depth(self, disparity):
        """Convert disparity map to depth map"""
        # Avoid division by zero
        mask = disparity > 0
        depth = np.zeros_like(disparity)
        
        # Use the Q matrix for accurate depth computation
        # Depth = (focal_length * baseline) / disparity
        focal_length = self.Q[2, 3]  # -P2[0,0] from Q matrix
        depth[mask] = -focal_length / disparity[mask]
        
        return depth, mask
    
    def filter_depth(self, depth, mask, min_depth=0.1, max_depth=10.0):
        """Filter depth values to reasonable range"""
        valid_mask = mask & (depth > min_depth) & (depth < max_depth)
        filtered_depth = depth.copy()
        filtered_depth[~valid_mask] = 0
        
        return filtered_depth, valid_mask
    
    def create_colored_depth_map(self, depth, mask):
        """Create a colored visualization of the depth map"""
        # Normalize depth for visualization
        valid_depth = depth[mask]
        if len(valid_depth) == 0:
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        
        depth_norm = np.zeros_like(depth)
        min_depth, max_depth = valid_depth.min(), valid_depth.max()
        
        if max_depth > min_depth:
            depth_norm[mask] = (depth[mask] - min_depth) / (max_depth - min_depth)
        
        # Apply colormap
        colored_depth = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Set invalid regions to black
        colored_depth[~mask] = [0, 0, 0]
        
        return colored_depth
    
    def switch_matcher(self):
        """Switch between different stereo matchers"""
        if self.has_wls:
            matchers = ['bm', 'sgbm', 'wls']
        else:
            matchers = ['bm', 'sgbm']
        
        current_idx = matchers.index(self.current_matcher) if self.current_matcher in matchers else 0
        self.current_matcher = matchers[(current_idx + 1) % len(matchers)]
        print(f"Switched to {self.current_matcher.upper()} matcher")

def main():
    # GStreamer pipelines for your cameras
    pipeline_cam0 = (
        "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36 ! "
        "video/x-raw,format=BGR,width=1280,height=960,framerate=30/1 ! "
        "videoconvert ! appsink drop=1 max-buffers=1"
    )

    pipeline_cam1 = (
        "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36 ! "
        "video/x-raw,format=BGR,width=1280,height=960,framerate=30/1 ! "
        "videoconvert ! appsink drop=1 max-buffers=1"
    )

    cap1 = cv2.VideoCapture(pipeline_cam0, cv2.CAP_GSTREAMER)
    cap2 = cv2.VideoCapture(pipeline_cam1, cv2.CAP_GSTREAMER)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open cameras")
        return
    
    # Initialize dense depth estimator
    depth_estimator = DenseDepthEstimator()
    
    print("Controls:")
    print("'q' - Quit")
    print("'s' - Save depth map")
    print("'m' - Switch matcher (BM/SGBM" + ("/WLS" if depth_estimator.has_wls else "") + ")")
    print("'r' - Show rectified images")
    print("'d' - Show disparity map")
    print("'c' - Show colored depth map")
    
    show_rectified = False
    show_disparity = True
    show_colored = True
    
    while True:
        # Capture frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: Could not read frames")
            break
        
        # Rectify images
        img1_rect, img2_rect = depth_estimator.rectify_images(frame1, frame2)
        
        # Compute disparity
        disparity = depth_estimator.compute_disparity(img1_rect, img2_rect)
        
        # Convert to depth
        depth, mask = depth_estimator.disparity_to_depth(disparity)
        
        # Filter depth
        depth_filtered, valid_mask = depth_estimator.filter_depth(depth, mask)
        
        # Create visualizations
        if show_colored:
            colored_depth = depth_estimator.create_colored_depth_map(depth_filtered, valid_mask)
            cv2.imshow('Colored Depth Map', colored_depth)
        
        if show_disparity:
            # Normalize disparity for display
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow('Disparity Map', disp_vis)
        
        if show_rectified:
            # Show rectified images side by side
            rectified_pair = np.hstack([img1_rect, img2_rect])
            # Add horizontal lines to show rectification
            h, w = rectified_pair.shape[:2]
            for i in range(0, h, 50):
                cv2.line(rectified_pair, (0, i), (w, i), (0, 255, 0), 1)
            cv2.imshow('Rectified Stereo Pair', rectified_pair)
        
        # Show original images
        original_pair = np.hstack([frame1, frame2])
        cv2.imshow('Original Stereo Pair', original_pair)
        
        # Display statistics
        valid_pixels = np.sum(valid_mask)
        total_pixels = depth.shape[0] * depth.shape[1]
        if valid_pixels > 0:
            avg_depth = np.mean(depth_filtered[valid_mask])
            min_depth = np.min(depth_filtered[valid_mask])
            max_depth = np.max(depth_filtered[valid_mask])
            coverage = (valid_pixels / total_pixels) * 100
            
            print(f"Matcher: {depth_estimator.current_matcher.upper()} | "
                  f"Coverage: {coverage:.1f}% | "
                  f"Depth: {min_depth:.2f}-{max_depth:.2f}m (avg: {avg_depth:.2f}m)")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save depth map and related data
            timestamp = cv2.getTickCount()
            cv2.imwrite(f'depth_map_{timestamp}.png', (depth_filtered * 1000).astype(np.uint16))
            cv2.imwrite(f'colored_depth_{timestamp}.png', colored_depth)
            cv2.imwrite(f'disparity_{timestamp}.png', disp_vis)
            np.save(f'depth_array_{timestamp}.npy', depth_filtered)
            print(f"Saved depth data with timestamp {timestamp}")
        elif key == ord('m'):
            depth_estimator.switch_matcher()
        elif key == ord('r'):
            show_rectified = not show_rectified
            if not show_rectified:
                cv2.destroyWindow('Rectified Stereo Pair')
        elif key == ord('d'):
            show_disparity = not show_disparity
            if not show_disparity:
                cv2.destroyWindow('Disparity Map')
        elif key == ord('c'):
            show_colored = not show_colored
            if not show_colored:
                cv2.destroyWindow('Colored Depth Map')
    
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()