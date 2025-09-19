import cv2
import numpy as np
import os
import argparse
from typing import NamedTuple, Any
from time import strftime, time
from pathlib import Path

from erics_cameras.usb_cam import USBCam
from erics_cameras.stereo_cam import StereoCam

CALIB_BATCH_SIZE = 15

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stereo camera calibration script using fisheye model with USB camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stereo calibration with USB camera (default)
  python stereo_calibrate.py
  
  # Specify USB device path
  python stereo_calibrate.py --video_path /dev/video0
  
  # Disable motion and time checks for faster processing
  python stereo_calibrate.py --disable_motion_check --disable_time_check
        """
    )
    
    parser.add_argument(
        "--video_path", 
        help="Path to USB video device", 
        type=str, 
        default="/dev/video0"
    )
    parser.add_argument(
        "--disable_motion_check", 
        help="Disable motion check between consecutive images", 
        action="store_true"
    )
    parser.add_argument(
        "--disable_time_check", 
        help="Disable time check between consecutive images", 
        action="store_true"
    )
    
    args = parser.parse_args()

    class BoardDetectionResults(NamedTuple):
        charuco_corners: Any
        charuco_ids: Any
        aruco_corners: Any
        aruco_ids: Any

    class PointReferences(NamedTuple):
        object_points: Any
        image_points: Any

    class StereoCalibrationResults(NamedTuple):
        repError: float
        K1: Any
        D1: Any
        K2: Any
        D2: Any
        R: Any
        T: Any
        rvecs: Any
        tvecs: Any

    # ChArUco board configuration
    SQUARE_LENGTH = 500
    MARKER_LENGTH = 300
    NUMBER_OF_SQUARES_VERTICALLY = 11
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=charuco_marker_dictionary,
    )

    # Initial camera matrices (will be refined during calibration)

    cam_mat1 = np.array([[492.13911009,0.,619.74260304],[0.,492.22808195,420.97601015],[0.,0.,1.,]])
    cam_mat2 = np.array([[492.13911009,0.,619.74260304],[0.,492.22808195,420.97601015],[0.,0.,1.,]])

    # Initial distortion coefficients for fisheye model (k1, k2, k3, k4)
    dist_coeffs1 = np.array([[0.04272378], [-0.01961093], [-0.00135352], [0.00050177]])
    dist_coeffs2 = np.array([[0.04272378], [-0.01961093], [-0.00135352], [0.00050177]])
    
    DIM = (1280, 720)  # Half width for each camera from dual camera setup

    # Initialize stereo camera
    usb_cam = USBCam(
        log_dir="./testimages",
        resolution=USBCam.ResolutionOption.R720P_DUAL,
        video_path=args.video_path
    )
    stereo_cam = StereoCam(usb_cam=usb_cam)
    left_cam = stereo_cam.get_left_camera()
    right_cam = stereo_cam.get_right_camera()

    # Create logs directory
    logs_base = Path("logs")
    time_dir = Path(strftime("%Y-%m-%d/%H-%M"))
    logs_path = logs_base / time_dir
    imgs_path = logs_path / "stereo_calib_imgs"
    imgs_path.mkdir(exist_ok=True, parents=True)
    
    # Create visualization windows
    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("right", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("charuco_board", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("left", (640, 360))
    cv2.resizeWindow("right", (640, 360))
    
    # Generate and display board image
    board_img = cv2.cvtColor(
        cv2.rotate(charuco_board.generateImage((1080, 1920), marginSize=10), cv2.ROTATE_90_CLOCKWISE), 
        cv2.COLOR_GRAY2BGR
    )

    # Storage for calibration data
    total_object_points = []
    total_image_points_left = []
    total_image_points_right = []
    num_total_images_used = 0
    last_image_add_time = time()

    # Pose tracking for motion detection
    pose_circular_buffer_left = np.empty((100, 6), dtype=np.float32)
    pose_circular_buffer_right = np.empty((100, 6), dtype=np.float32)
    pose_circular_buffer_index = 0
    pose_circular_buffer_size = 0

    last_detection_results_left = None
    last_detection_results_right = None

    def run_stereo_calibration(sample_indices: list[int]):
        global cam_mat1, cam_mat2, dist_coeffs1, dist_coeffs2
        print(f"Running stereo fisheye calibration with {len(sample_indices)} samples")
        
        if len(total_object_points) == 0 or len(total_image_points_left) == 0 or len(total_image_points_right) == 0:
            print("No valid points collected. Cannot perform calibration.")
            return
        
        object_points_list = [total_object_points[i] for i in sample_indices]
        image_points_left_list = [total_image_points_left[i] for i in sample_indices]
        image_points_right_list = [total_image_points_right[i] for i in sample_indices]
        
        # Stereo fisheye calibration
        calibration_results = StereoCalibrationResults(
            *cv2.fisheye.stereoCalibrate(
                object_points_list,
                image_points_left_list,
                image_points_right_list,
                cam_mat1,
                dist_coeffs1,
                cam_mat2,
                dist_coeffs2,
                DIM,
                None,  # R output
                None,  # T output
                flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        )
        
        print(f'Stereo reproj error: {calibration_results.repError:.4f}')
        
        # Format output
        matrix_outputs = '\n'.join([
            '# Left camera',
            f'K1 = np.array({",".join(str(calibration_results.K1).split())})'.replace('[,','['),
            f'D1 = np.array({",".join(str(calibration_results.D1).split())})'.replace('[,','['),
            '# Right camera', 
            f'K2 = np.array({",".join(str(calibration_results.K2).split())})'.replace('[,','['),
            f'D2 = np.array({",".join(str(calibration_results.D2).split())})'.replace('[,','['),
            '# Stereo parameters',
            f'R = np.array({",".join(str(calibration_results.R).split())})'.replace('[,','['),
            f'T = np.array({",".join(str(calibration_results.T).split())})'.replace('[,','[')
        ])
        print(matrix_outputs)
        
        # Save calibration results
        num_batches = num_total_images_used // CALIB_BATCH_SIZE
        calib_path = logs_path / 'stereo_intrinsics'
        calib_path.mkdir(exist_ok=True, parents=True)

        with open(f'{calib_path}/{num_batches}.txt', 'w+') as f:
            f.write(matrix_outputs)
        
        # Update camera matrices for next iteration
        cam_mat1 = calibration_results.K1
        cam_mat2 = calibration_results.K2
        dist_coeffs1 = calibration_results.D1
        dist_coeffs2 = calibration_results.D2
        
        return calibration_results

    print("Starting stereo calibration. Press 'q' to quit.")
    
    while True:
        # Get synchronized stereo image pair
        try:
            left_img, right_img = stereo_cam.get_image_pair()
            left_bgr = left_img.get_array()
            right_bgr = right_img.get_array()
        except RuntimeError as e:
            print(f"Failed to get stereo images: {e}")
            break

        # Convert to grayscale for detection
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        # Create debug images
        left_debug = left_bgr.copy()
        right_debug = right_bgr.copy()

        # Detect ChArUco board in both images
        charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
        left_detection = BoardDetectionResults(*charuco_detector.detectBoard(left_gray))
        right_detection = BoardDetectionResults(*charuco_detector.detectBoard(right_gray))

        left_reproj_err = None
        right_reproj_err = None
        left_pose_dist = None
        right_pose_dist = None

        # Check if we have valid detections in both images
        valid_left = (left_detection.charuco_corners is not None and 
                     len(left_detection.charuco_corners) > 4)
        valid_right = (right_detection.charuco_corners is not None and 
                      len(right_detection.charuco_corners) > 4)

        do_skip_pose = True
        
        if valid_left and valid_right:
            # Process left image
            left_point_refs = PointReferences(
                *charuco_board.matchImagePoints(
                    left_detection.charuco_corners, left_detection.charuco_ids
                )
            )
            
            # Process right image
            right_point_refs = PointReferences(
                *charuco_board.matchImagePoints(
                    right_detection.charuco_corners, right_detection.charuco_ids
                )
            )

            # Calculate poses and reprojection errors
            ret_left, rvecs_left, tvecs_left = cv2.fisheye.solvePnP(
                left_point_refs.object_points,
                left_point_refs.image_points.reshape(-1, 1, 2).astype(np.float32),
                cam_mat1,
                dist_coeffs1,
                flags=cv2.SOLVEPNP_IPPE,
            )
            
            ret_right, rvecs_right, tvecs_right = cv2.fisheye.solvePnP(
                right_point_refs.object_points,
                right_point_refs.image_points.reshape(-1, 1, 2).astype(np.float32),
                cam_mat2,
                dist_coeffs2,
                flags=cv2.SOLVEPNP_IPPE,
            )

            if ret_left and ret_right:
                # Calculate reprojection errors
                reproj_left = cv2.fisheye.projectPoints(
                    left_point_refs.object_points, rvecs_left, tvecs_left, cam_mat1, dist_coeffs1
                )[0].squeeze()
                
                reproj_right = cv2.fisheye.projectPoints(
                    right_point_refs.object_points, rvecs_right, tvecs_right, cam_mat2, dist_coeffs2
                )[0].squeeze()

                left_reproj_err = np.mean(
                    np.linalg.norm(left_point_refs.image_points.squeeze() - reproj_left, axis=1)
                )
                
                right_reproj_err = np.mean(
                    np.linalg.norm(right_point_refs.image_points.squeeze() - reproj_right, axis=1)
                )

                # Motion detection
                left_movement = 1e9
                right_movement = 1e9
                
                if not args.disable_motion_check and last_detection_results_left is not None:
                    # Check left camera movement
                    current_ids_left = left_detection.charuco_ids.squeeze().tolist()
                    last_ids_left = last_detection_results_left.charuco_ids.squeeze().tolist()
                    intersecting_ids_left = [i for i in current_ids_left if i in last_ids_left]
                    
                    if len(intersecting_ids_left) > 2:
                        current_intersect_left = np.array([
                            corner for id, corner in zip(current_ids_left, left_detection.charuco_corners) 
                            if id in last_ids_left
                        ])
                        last_intersect_left = np.array([
                            corner for id, corner in zip(last_ids_left, last_detection_results_left.charuco_corners) 
                            if id in current_ids_left
                        ])
                        left_movement = np.mean(np.linalg.norm(
                            current_intersect_left.squeeze() - last_intersect_left.squeeze(), axis=1
                        ))

                if not args.disable_motion_check and last_detection_results_right is not None:
                    # Check right camera movement
                    current_ids_right = right_detection.charuco_ids.squeeze().tolist()
                    last_ids_right = last_detection_results_right.charuco_ids.squeeze().tolist()
                    intersecting_ids_right = [i for i in current_ids_right if i in last_ids_right]
                    
                    if len(intersecting_ids_right) > 2:
                        current_intersect_right = np.array([
                            corner for id, corner in zip(current_ids_right, right_detection.charuco_corners) 
                            if id in last_ids_right
                        ])
                        last_intersect_right = np.array([
                            corner for id, corner in zip(last_ids_right, last_detection_results_right.charuco_corners) 
                            if id in current_ids_right
                        ])
                        right_movement = np.mean(np.linalg.norm(
                            current_intersect_right.squeeze() - last_intersect_right.squeeze(), axis=1
                        ))

                last_detection_results_left = left_detection
                last_detection_results_right = right_detection

                # Pose uniqueness check
                if rvecs_left is not None and tvecs_left is not None and rvecs_right is not None and tvecs_right is not None:
                    combo_vec_left = np.concatenate((rvecs_left.squeeze(), tvecs_left.squeeze()))
                    combo_vec_right = np.concatenate((rvecs_right.squeeze(), tvecs_right.squeeze()))
                    
                    pose_too_close_left = (pose_circular_buffer_size > 0 and 
                                         (left_pose_dist := np.min(np.linalg.norm(
                                             pose_circular_buffer_left[:pose_circular_buffer_size] - combo_vec_left.reshape((1,6)), axis=1))) < 500)
                    
                    pose_too_close_right = (pose_circular_buffer_size > 0 and 
                                          (right_pose_dist := np.min(np.linalg.norm(
                                              pose_circular_buffer_right[:pose_circular_buffer_size] - combo_vec_right.reshape((1,6)), axis=1))) < 500)

                    motion_too_much = (not args.disable_motion_check and 
                                     (left_movement > 1 or right_movement > 1))
                    
                    time_too_recent = (not args.disable_time_check and 
                                     time() - last_image_add_time < 0.5)

                    if not (pose_too_close_left or pose_too_close_right or motion_too_much or time_too_recent):
                        pose_circular_buffer_left[pose_circular_buffer_index] = combo_vec_left
                        pose_circular_buffer_right[pose_circular_buffer_index] = combo_vec_right
                        pose_circular_buffer_index = (pose_circular_buffer_index + 1) % pose_circular_buffer_left.shape[0]
                        pose_circular_buffer_size = min(pose_circular_buffer_size + 1, pose_circular_buffer_left.shape[0])
                        do_skip_pose = False

                # Draw detection points
                for pt in left_point_refs.image_points.squeeze():
                    green_amount = int((1-np.tanh(4*(left_movement-1.5)))/4 * 255) if left_movement > 1 else 255
                    cv2.circle(left_debug, tuple(pt.astype(int)), 7, (255, green_amount, 0), -1)
                
                for pt in right_point_refs.image_points.squeeze():
                    green_amount = int((1-np.tanh(4*(right_movement-1.5)))/4 * 255) if right_movement > 1 else 255
                    cv2.circle(right_debug, tuple(pt.astype(int)), 7, (255, green_amount, 0), -1)

                # Draw reprojection points
                for pt in reproj_left:
                    if np.any(np.isnan(pt)) or np.any(pt < 0):
                        continue
                    try:
                        color = (0, 0, 255) if left_reproj_err > 1 else (0, 255, 0)
                        cv2.circle(left_debug, tuple(pt.astype(int)), 5, color, -1)
                    except:
                        print("Error drawing left reprojection circle")

                for pt in reproj_right:
                    if np.any(np.isnan(pt)) or np.any(pt < 0):
                        continue
                    try:
                        color = (0, 0, 255) if right_reproj_err > 1 else (0, 255, 0)
                        cv2.circle(right_debug, tuple(pt.astype(int)), 5, color, -1)
                    except:
                        print("Error drawing right reprojection circle")

        # Add text overlays
        text_color = (255, 120, 0)
        for img, reproj_err, pose_dist in [(left_debug, left_reproj_err, left_pose_dist), 
                                          (right_debug, right_reproj_err, right_pose_dist),
                                          (board_img, None, None)]:
            cv2.rectangle(img, (0, 0), (280, 60), (0, 0, 0), -1)
            
            if reproj_err is not None:
                color = (0, 255, 0) if reproj_err < 1 else (0, 0, 255)
                cv2.putText(img, f"Reproj Err: {reproj_err:.2f}", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(img, "Reproj Err: N/A", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            cv2.putText(img, f"N good stereo pairs: {num_total_images_used}", (5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if pose_dist is not None:
                cv2.putText(img, f"Originality: {pose_dist/500:.2f}", (5, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(img, "Originality: N/A", (5, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display images
        cv2.imshow("left", cv2.resize(left_debug, (640, 360)))
        cv2.imshow("right", cv2.resize(right_debug, (640, 360)))
        cv2.imshow("charuco_board", board_img)
        
        key = cv2.waitKey(1)

        # Add image pair to calibration dataset
        if (not do_skip_pose and valid_left and valid_right and 
            left_reproj_err is not None and left_reproj_err > 1 and
            right_reproj_err is not None and right_reproj_err > 1 and
            len(left_point_refs.object_points) > 4 and len(right_point_refs.object_points) > 4):
            
            total_object_points.append(left_point_refs.object_points)  # Same for both cameras
            total_image_points_left.append(left_point_refs.image_points)
            total_image_points_right.append(right_point_refs.image_points)
            num_total_images_used += 1
            last_image_add_time = time()
            
            # Save images
            cv2.imwrite(f'{imgs_path}/left_{len(list(imgs_path.glob("left_*.png")))}.png', left_bgr)
            cv2.imwrite(f'{imgs_path}/right_{len(list(imgs_path.glob("right_*.png")))}.png', right_bgr)

            # Run calibration periodically
            is_time_to_calib = num_total_images_used % CALIB_BATCH_SIZE == 0
            if num_total_images_used >= CALIB_BATCH_SIZE and is_time_to_calib:
                sample_indices = np.random.choice(
                    np.arange(num_total_images_used), 
                    min(60, num_total_images_used), 
                    replace=False
                )
                run_stereo_calibration(sample_indices)

        if key == ord("q"):
            break

    # Final calibration with all data
    if num_total_images_used > 0:
        print(f"\nRunning final stereo calibration with all {num_total_images_used} image pairs...")
        final_results = run_stereo_calibration(np.arange(num_total_images_used))
        
        if final_results:
            # Compute stereo rectification
            print("\nComputing stereo rectification...")
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                final_results.K1, final_results.D1,
                final_results.K2, final_results.D2,
                DIM,
                final_results.R, final_results.T,
                flags=0,
                newImageSize=DIM,
                balance=0.0,
                fov_scale=1.0
            )
            
            # Save rectification parameters
            rect_path = logs_path / 'rectification'
            rect_path.mkdir(exist_ok=True, parents=True)
            
            rect_outputs = '\n'.join([
                '# Rectification matrices',
                f'R1 = np.array({",".join(str(R1).split())})'.replace('[,','['),
                f'R2 = np.array({",".join(str(R2).split())})'.replace('[,','['),
                f'P1 = np.array({",".join(str(P1).split())})'.replace('[,','['),
                f'P2 = np.array({",".join(str(P2).split())})'.replace('[,','['),
                f'Q = np.array({",".join(str(Q).split())})'.replace('[,','['),
            ])
            
            with open(f'{rect_path}/rectification.txt', 'w+') as f:
                f.write(rect_outputs)
            
            print("Rectification parameters saved.")
            print(rect_outputs)

    print(f"\nStereo calibration complete! Results saved to {logs_path}")
    cv2.destroyAllWindows()