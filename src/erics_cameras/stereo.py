import cv2
import numpy as np
import matplotlib.pyplot as plt

R = np.eye(3)
t = [0.116,0,0]

cam_mat = np.array([[297.80062345,0.,685.72493754],[0.,298.63865273,451.61133244],[0.,0.,1.,]])
# k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, Tx, Ty
dist_coeffs = np.array([[-0.20148179,0.03270111,0.,0.,-0.00211291]])

class StereoDepthEstimator:
    def __init__(self):
        # Hardcoded camera intrinsics (modify these with your actual calibration data)
        self.K1 = cam_mat
        self.K2 = cam_mat
        
        # Distortion coefficients (k1, k2, p1, p2, k3)
        self.dist1 = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
        self.dist2 = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
        
        # Extrinsic parameters (Rotation and Translation from camera 1 to camera 2)
        # These represent the transformation from left camera to right camera
        self.R = R
        
        self.T = np.array(t).reshape((3,1))
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Initialize FLANN matcher for better performance
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Projection matrices
        self.P1 = np.hstack([self.K1, np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])
    
    def find_and_match_features(self, img1, img2):
        """Find ORB features and match them between two images"""
        # Find keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return [], [], []
        
        # Match features using FLANN
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2, good_matches
    
    def triangulate_points(self, pts1, pts2):
        """Triangulate 3D points from matched 2D points"""
        if len(pts1) == 0 or len(pts2) == 0:
            return np.array([])
        
        # Undistort points
        pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K1, self.dist1, P=self.K1)
        pts2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K2, self.dist2, P=self.K2)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                         pts1_undist.reshape(-1, 2).T,
                                         pts2_undist.reshape(-1, 2).T)
        
        # Convert from homogeneous coordinates
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def compute_depth(self, points_3d):
        """Compute depth (Z coordinate) from 3D points"""
        if len(points_3d) == 0:
            return np.array([])
        return points_3d[:, 2]
    
    def filter_valid_depths(self, points_3d, pts1, min_depth=0.1, max_depth=50.0):
        """Filter out invalid depth measurements"""
        if len(points_3d) == 0:
            return np.array([]), np.array([])
        
        depths = self.compute_depth(points_3d)
        valid_mask = (depths > min_depth) & (depths < max_depth) & (depths > 0)
        
        return points_3d[valid_mask], pts1[valid_mask]
    
    def visualize_depth(self, img1, pts1, depths):
        """Visualize depth information on the image"""
        if len(depths) == 0:
            return img1
        
        img_depth = img1.copy()
        
        # Normalize depths for color mapping
        if len(depths) > 0:
            depth_norm = (depths - depths.min()) / (depths.max() - depths.min())
            
            for i, (pt, depth, depth_n) in enumerate(zip(pts1, depths, depth_norm)):
                # Color based on depth (closer = red, farther = blue)
                color = (int(255 * (1 - depth_n)), int(255 * depth_n * 0.5), int(255 * depth_n))
                cv2.circle(img_depth, (int(pt[0]), int(pt[1])), 3, color, -1)
                
                # Add depth text
                cv2.putText(img_depth, f'{depth:.2f}m', 
                           (int(pt[0]) + 5, int(pt[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return img_depth

def main():
    # Initialize cameras (modify these indices based on your setup)
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
    
    # Set camera properties (optional)
    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open cameras")
        return
    
    # Initialize depth estimator
    depth_estimator = StereoDepthEstimator()
    
    print("Press 'q' to quit, 's' to save current depth data")
    
    while True:
        # Capture frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: Could not read frames")
            break
        
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Find and match features
        pts1, pts2, matches = depth_estimator.find_and_match_features(gray1, gray2)
        
        if len(pts1) > 0:
            # Triangulate 3D points
            points_3d = depth_estimator.triangulate_points(pts1, pts2)
            
            # Filter valid depths
            valid_points_3d, valid_pts1 = depth_estimator.filter_valid_depths(points_3d, pts1)
            
            if len(valid_points_3d) > 0:
                # Compute depths
                depths = depth_estimator.compute_depth(valid_points_3d)
                
                # Visualize results
                img_with_depth = depth_estimator.visualize_depth(frame1, valid_pts1, depths)
                
                # Display statistics
                cv2.putText(img_with_depth, f'Features: {len(valid_pts1)}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_with_depth, f'Avg Depth: {depths.mean():.2f}m', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_with_depth, f'Min/Max: {depths.min():.2f}/{depths.max():.2f}m', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display images
                cv2.imshow('Left Camera with Depth', img_with_depth)
                cv2.imshow('Right Camera', frame2)
                
                # Print depth statistics
                print(f"Found {len(valid_pts1)} valid features, "
                      f"Depth range: {depths.min():.2f} - {depths.max():.2f}m, "
                      f"Average: {depths.mean():.2f}m")
            else:
                cv2.imshow('Left Camera with Depth', frame1)
                cv2.imshow('Right Camera', frame2)
                print("No valid depth measurements found")
        else:
            cv2.imshow('Left Camera with Depth', frame1)
            cv2.imshow('Right Camera', frame2)
            print("No feature matches found")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(pts1) > 0:
            # Save current depth data
            points_3d = depth_estimator.triangulate_points(pts1, pts2)
            valid_points_3d, valid_pts1 = depth_estimator.filter_valid_depths(points_3d, pts1)
            if len(valid_points_3d) > 0:
                np.save('depth_points.npy', valid_points_3d)
                np.save('image_points.npy', valid_pts1)
                print(f"Saved {len(valid_points_3d)} depth points to files")
    
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()