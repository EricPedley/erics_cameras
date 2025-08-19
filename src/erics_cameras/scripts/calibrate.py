import cv2

from typing import NamedTuple

import numpy as np
import os

from erics_cameras import GstCamera, ReplayCamera, CSICamera, USBCamera, LibCameraCam
from time import strftime, time
from pathlib import Path
from typing import Any

import argparse
CALIB_BATCH_SIZE = 15

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Camera calibration script with support for live camera, image folders, and video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live camera calibration (default)
  python calibrate.py
  
  # Calibrate from image folder
  python calibrate.py --source folder --source_path /path/to/images
  
  # Calibrate from video file
  python calibrate.py --source video --source_path /path/to/video.mp4
  
          # Disable motion and time checks for faster processing
        python calibrate.py --source folder --source_path /path/to/images --disable_motion_check --disable_time_check
        
        # Enable visualizations when using folder/video source
        python calibrate.py --source folder --source_path /path/to/images --force_visualization
        
        # Manual exposure control for live camera
        python calibrate.py --exposure_mode manual --exposure_time 5000 --gain_mode manual --analogue_gain 2.5
        
        # Auto exposure with EV adjustment
        python calibrate.py --exposure_mode auto --exposure_value -1.0 --brightness 0.2 --contrast 1.2
        """
    )
    acceptable_sources = [
        "folder",
        "video",
        "v4l2",
        "nvargus",
        "libcamera"
    ]
    parser.add_argument(
        "--source", help=f"Source for calibration: {', '.join(acceptable_sources)}. If 'folder' or 'video', provide the path.", 
        choices=acceptable_sources
    )
    parser.add_argument(
        "--source_path", help="Path to image folder or video file when using --source folder or video", 
        type=str, default=None
    )
    parser.add_argument(
        "--disable_motion_check", help="Disable motion check between consecutive images", 
        action="store_true"
    )
    parser.add_argument(
        "--disable_time_check", help="Disable time check between consecutive images", 
        action="store_true"
    )
    parser.add_argument(
        "--calibration_model", help="Calibration model to use: 'normal' (k1,k2,p1,p2,k3) or 'fisheye' (k1,k2,p1,p2)", 
        choices=["normal", "fisheye"], default="fisheye"
    )
    parser.add_argument(
        "--force_visualization", help="Force enable visualizations even when using folder/video source", 
        action="store_true"
    )
    parser.add_argument(
        "--exposure_mode", help="Exposure mode: 'auto' or 'manual'", 
        choices=["auto", "manual"], default="auto"
    )

    parser.add_argument(
        "--exposure_value", help="Exposure value (EV) adjustment for auto exposure",
        type=int, default=10000
    )

    parser.add_argument(
        "--gain_mode", help="Gain mode: 'auto' or 'manual'", 
        choices=["auto", "manual"], default="auto"
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


    class CameraCalibrationResults(NamedTuple):
        repError: float
        camMatrix: Any
        distcoeff: Any
        rvecs: Any
        tvecs: Any


    SQUARE_LENGTH = 500
    MARKER_LENGHT = 300
    NUMBER_OF_SQUARES_VERTICALLY = 11
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGHT,
        dictionary=charuco_marker_dictionary,
    )

    cam_mat = np.array([[266.61728276,0.,643.83126137],[0.,266.94450686,494.81811813],[0.,0.,1.,]])
    # k1, k2, p1, p2, k3 (normal model)
    DIM = (1280, 960)
    
    # Initialize distortion coefficients based on calibration model
    if args.calibration_model == "fisheye":
        dist_coeffs = np.zeros((1,4), dtype=np.float32)  # Fisheye uses 4 distortion coefficients (k1, k2, p1, p2)
    else:
        dist_coeffs = np.array([[-6.07417419e-02,9.95447444e-02,-2.26448001e-04,1.22881804e-03,3.42134205e-03,1.45361886e-01,8.03248099e-02,2.11170107e-02,-3.80620047e-03,2.48350591e-05,-8.33565666e-04,2.97806723e-05]])
    
    # Initialize undistortion maps
    map1, map2 = None, None

    # Set calibration functions based on model
    if args.calibration_model == "fisheye":
        calibrate_function = cv2.fisheye.calibrate
        pnp_function = cv2.fisheye.solvePnP
        project_function = cv2.fisheye.projectPoints
    else:
        calibrate_function = cv2.calibrateCamera
        pnp_function = cv2.solvePnP
        project_function = cv2.projectPoints

    total_object_points = []
    total_image_points = []
    num_total_images_used = 0
    last_image_add_time = time()

    # Determine source type and initialize camera
    if args.source in ["v4l2", "nvargus", "libcamera"]:
        LIVE = True
        logs_base = Path("logs")
        time_dir = Path(strftime("%Y-%m-%d/%H-%M"))
        logs_path = logs_base / time_dir

        if args.source == "libcamera":
            camera = LibCameraCam(
                log_dir="./testimages",
                resolution=LibCameraCam.ResolutionOption.R720P,
                camera_name="/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36",
                framerate=30,
                exposure_mode=LibCameraCam.ExposureMode(args.exposure_mode),
            )
        elif args.source == "nvargus":
            camera = CSICamera(None)
        elif args.source == "v4l2":
            camera = USBCamera(None)
            
            # Create control window with sliders for camera properties
            class CameraController:
                def __init__(self, camera):
                    self.camera = camera
                    self.prev_brightness = 50
                    self.prev_contrast = 50
                    self.prev_saturation = 50
                    self.prev_hue = 50
                    
                    def nothing(x):
                        pass
                        
                    cv2.namedWindow("Camera Controls", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Camera Controls", (400, 300))
                    
                    # Create trackbars for camera properties
                    cv2.createTrackbar("Brightness", "Camera Controls", 0, 100, nothing)
                    cv2.createTrackbar("Contrast", "Camera Controls", 0, 100, nothing)
                    cv2.createTrackbar("Saturation", "Camera Controls", 0, 100, nothing)
                    cv2.createTrackbar("Hue", "Camera Controls", 0, 100, nothing)
                    
                    # Set initial values to middle (50)
                    cv2.setTrackbarPos("Brightness", "Camera Controls", 50)
                    cv2.setTrackbarPos("Contrast", "Camera Controls", 50)
                    cv2.setTrackbarPos("Saturation", "Camera Controls", 50)
                    cv2.setTrackbarPos("Hue", "Camera Controls", 50)
                
                def update_properties(self):
                    brightness = cv2.getTrackbarPos("Brightness", "Camera Controls")
                    contrast = cv2.getTrackbarPos("Contrast", "Camera Controls")
                    saturation = cv2.getTrackbarPos("Saturation", "Camera Controls")
                    hue = cv2.getTrackbarPos("Hue", "Camera Controls")
                    
                    # Check if any value changed
                    if (brightness != self.prev_brightness or contrast != self.prev_contrast or 
                        saturation != self.prev_saturation or hue != self.prev_hue):
                        
                        # Convert 0-100 range to v4l2 range (-2147483648 to 2147483647)
                        # Map 0-100 to -100 to 100 for reasonable values
                        brightness_v4l2 = int((brightness - 50) * 2)
                        contrast_v4l2 = int((contrast - 50) * 2)
                        saturation_v4l2 = int((saturation - 50) * 2)
                        hue_v4l2 = int((hue - 50) * 2)
                        
                        print(f"Updating camera properties: B={brightness_v4l2}, C={contrast_v4l2}, S={saturation_v4l2}, H={hue_v4l2}")
                        
                        # Update camera properties
                        success = self.camera.update_properties(
                            brightness=brightness_v4l2,
                            contrast=contrast_v4l2,
                            saturation=saturation_v4l2,
                            hue=hue_v4l2
                        )
                        
                        if success:
                            self.prev_brightness = brightness
                            self.prev_contrast = contrast
                            self.prev_saturation = saturation
                            self.prev_hue = hue
                        else:
                            print("Failed to update camera properties")
            
            # Create camera controller
            camera_controller = CameraController(camera)
        cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
        cv2.namedWindow("charuco_board", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("calib", (1024, 576))
        board_img = cv2.cvtColor(cv2.rotate(charuco_board.generateImage((1080,1920), marginSize=10), cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_GRAY2BGR)
        # cv2.resizeWindow("charuco_board", (1600,900))
        
        imgs_path = logs_path / "calib_imgs"
        imgs_path.mkdir(exist_ok=True, parents=True)
        
        # Display camera settings
        print("Camera settings:")
        print(f"  Resolution: {camera._resolution.value}")
        print(f"  Exposure mode: {args.exposure_mode}")
        if args.exposure_mode == "manual":
            print(f"  Exposure time: {args.exposure_time} μs")
        print(f"  Gain mode: {args.gain_mode}")
        if args.gain_mode == "manual":
            print(f"  Analogue gain: {args.analogue_gain}")
        print(f"  Exposure value: {args.exposure_value}")
        print(f"  Brightness: {args.brightness}")
        print(f"  Contrast: {args.contrast}")
        print(f"  Saturation: {args.saturation}")
        print(f"  Sharpness: {args.sharpness}")
        
    elif args.source in ["folder", "video"]:
        LIVE = False
        if args.source_path is None:
            raise ValueError(f"--source_path must be provided when using --source {args.source}")
        
        # Initialize ReplayCamera
        camera = ReplayCamera(args.source_path)
        print(f"Using ReplayCamera with {args.source}: {args.source_path}")
        print(f"Total frames: {camera.get_total_frames()}")
        
        # Display additional info for video files
        if args.source == "video":
            video_info = camera.get_video_info()
            if video_info:
                print(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f} fps")
                if video_info['duration_seconds']:
                    print(f"Duration: {video_info['duration_seconds']:.1f} seconds")
        
        # Display check status
        print(f"Motion check: {'enabled' if not args.disable_motion_check else 'disabled'}")
        print(f"Time check: {'enabled' if not args.disable_time_check else 'disabled'}")
        print(f"Visualization: {'enabled' if args.force_visualization else 'disabled'}")
        
        # Create logs directory for output
        logs_base = Path("logs")
        time_dir = Path(strftime("%Y-%m-%d/%H-%M"))
        logs_path = logs_base / time_dir
        imgs_path = logs_path / "calib_imgs"
        imgs_path.mkdir(exist_ok=True, parents=True)
        
        # Generate board image for display
        board_img = cv2.cvtColor(cv2.rotate(charuco_board.generateImage((1080,1920), marginSize=10), cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_GRAY2BGR)
        
        # Create visualization windows if forced
        if args.force_visualization:
            try:
                cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
                cv2.namedWindow("charuco_board", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("calib", (1024, 576))
                print("Visualization windows created successfully")
            except cv2.error as e:
                print(f"Warning: Could not create visualization windows: {e}")
                print("Visualization will be disabled (this is normal in headless environments)")
                args.force_visualization = False
        
    else:
        raise ValueError(f"Invalid source type: {args.source}")

    index = 0
    images = []  # Not used for ReplayCamera

    det_results: list[BoardDetectionResults] = []

    latest_error = None

    pose_circular_buffer = np.empty((100, 6), dtype=np.float32)
    pose_circular_buffer_index = 0
    pose_circular_buffer_size = 0

    last_detection_results = None

    while True:
        def run_calibration(sample_indices: list[int]):
            global cam_mat
            print(f"Running {args.calibration_model} calibration with {len(sample_indices)} samples")
            
            # Check if we have any valid points to calibrate with
            if len(total_object_points) == 0 or len(total_image_points) == 0:
                print("No valid object points or image points collected. Cannot perform calibration.")
                return
            
            object_points_list = [total_object_points[i] for i in sample_indices]
            image_points_list = [total_image_points[i] for i in sample_indices]
            
            # Use the appropriate calibration function
            if args.calibration_model == "fisheye":
                calibration_results = CameraCalibrationResults(
                    *calibrate_function(
                        object_points_list,
                        image_points_list,
                        shape,
                        cam_mat, 
                        np.zeros((4, 1)),
                        flags=cv2.fisheye.CALIB_FIX_SKEW,
                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
                )
                dist_coeffs_comment = "# k1, k2, k3, k4 (fisheye model)"
            else:
                calibration_results = CameraCalibrationResults(
                    *calibrate_function(
                        object_points_list,
                        image_points_list,
                        shape,
                        None,
                        None,
                        flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
                )
                dist_coeffs_comment = "# k1, k2, p1, p2, k3 (normal model)"

            print(f'Reproj error: {calibration_results.repError}')
            latest_error = calibration_results.repError
            matrix_outputs = '\n'.join([
                f'cam_mat = np.array({",".join(str(calibration_results.camMatrix).split())})'.replace('[,','['),
                dist_coeffs_comment,
                f'dist_coeffs = np.array({",".join(str(calibration_results.distcoeff).split())})'.replace('[,','[')
            ])
            print(matrix_outputs)
            
            num_batches = num_total_images_used//CALIB_BATCH_SIZE
            calib_path = logs_path / 'intrinsics'
            calib_path.mkdir(exist_ok=True, parents=True)

            with open(f'{calib_path}/{num_batches}.txt', 'w+') as f:
                f.write(matrix_outputs)
            cam_mat = calibration_results.camMatrix
            dist_coeffs = calibration_results.distcoeff

        if LIVE:
            try:
                img = camera.take_image()
            except RuntimeError as e:
                print(f"Live camera exhausted: {e}")
                break
            if img is None:
                print("Failed to get image")
                continue
            img_bgr = img.get_array()
            # ret, img_bgr = cap0.read()
            # cv2.imshow("debug", img_bgr")
        else:
            # Using ReplayCamera
            try:
                img = camera.take_image()
                if img is None:
                    print("Failed to get image from ReplayCamera")
                    break
                img_bgr = img.get_array()
                index += 1
                print(f"Processing frame {index}/{camera.get_total_frames()}")
            except RuntimeError as e:
                print(f"ReplayCamera exhausted: {e}")
                break

        img_debug = img_bgr.copy()

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        do_undistortion_trickery = dist_coeffs[0][0] != 0
        if do_undistortion_trickery:
            if args.calibration_model == "fisheye":
                img_gray_undistorted = cv2.remap(img_gray, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            else:
                img_gray_undistorted = cv2.undistort(img_gray, cam_mat, dist_coeffs)
        else:
            img_gray_undistorted = img_gray
        undistorted_debug = cv2.cvtColor(img_gray_undistorted, cv2.COLOR_GRAY2BGR)
        charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
        detection_results = BoardDetectionResults(*charuco_detector.detectBoard(img_gray))


        img_avg_reproj_err = None
        closest_pose_dist = None
        if (
            detection_results.charuco_corners is not None
            and len(detection_results.charuco_corners) > 4
        ):
            det_results.append(detection_results)
            point_references = PointReferences(
                *charuco_board.matchImagePoints(
                    detection_results.charuco_corners, detection_results.charuco_ids
                )
            )


            # Use the appropriate PnP and projection functions
            ret, rvecs, tvecs = pnp_function(
                point_references.object_points,
                point_references.image_points.reshape(-1, 1, 2).astype(np.float32),
                cam_mat,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ret:
                reproj: np.ndarray = project_function(
                    point_references.object_points, rvecs, tvecs, cam_mat, dist_coeffs
                )[0].squeeze()

                img_avg_reproj_err = np.mean(
                    np.linalg.norm(
                        point_references.image_points.squeeze() - reproj, axis=1
                    )
                )
                
                movement_magnitude=1e9
                if not args.disable_motion_check and last_detection_results is not None:
                    current_ids = [id for id in detection_results.charuco_ids.squeeze().tolist()]
                    last_ids = [id for id in last_detection_results.charuco_ids.squeeze().tolist()]
                    intersecting_ids = [i for i in current_ids if i in last_ids]
                    if len(intersecting_ids) > 2:
                        current_intersect_charuco_corners = np.array([
                            corner
                            for id, corner in zip(
                                current_ids,
                                detection_results.charuco_corners
                            ) if id in last_ids
                        ])

                    
                        last_intersect_charuco_corners = np.array([
                            corner
                            for id, corner in zip(
                                last_ids,
                                last_detection_results.charuco_corners
                            ) if id in current_ids
                        ])

                        current_intersecting_point_references = PointReferences(
                            *charuco_board.matchImagePoints(
                                current_intersect_charuco_corners, np.array(intersecting_ids).reshape((-1, 1))
                            )
                        )

                        last_intersection_point_references = PointReferences(
                            *charuco_board.matchImagePoints(
                                last_intersect_charuco_corners, np.array(intersecting_ids).reshape((-1, 1))
                            )
                        )

                        movement_magnitude = np.mean(np.linalg.norm(current_intersecting_point_references.image_points.squeeze() - last_intersection_point_references.image_points.squeeze(), axis=1))
                last_detection_results = detection_results

                for pt in point_references.image_points.squeeze():
                    green_amount = int((1-np.tanh(4*(movement_magnitude-1.5)))/4 *255) if movement_magnitude>1 else 255
                    cv2.circle(
                        img_debug, tuple(pt.astype(int)), 7, (255, green_amount, 0), -1
                    )
                for pt in reproj:
                    if np.any(np.isnan(pt)) or np.any(pt<0):
                        continue
                    try:
                        cv2.circle(img_debug, tuple(pt.astype(int)), 5,(0, 0, 255) if img_avg_reproj_err > 1 else (0,255,0),-1)
                    except:
                        print("Error in cv2 circle")


                
            if rvecs is None or tvecs is None :
                do_skip_pose = True
            else:
                combo_vec = np.concatenate((rvecs.squeeze(), tvecs.squeeze()))
                pose_too_close = pose_circular_buffer_size > 0 and (closest_pose_dist:=np.min(np.linalg.norm(pose_circular_buffer[:pose_circular_buffer_size] - combo_vec.reshape((1,6)), axis=1))) < 500
                if pose_too_close or (not args.disable_motion_check and movement_magnitude>1):
                    do_skip_pose = True
                else:
                    pose_circular_buffer[pose_circular_buffer_index] = combo_vec
                    pose_circular_buffer_index = (pose_circular_buffer_index + 1) % pose_circular_buffer.shape[0]
                    pose_circular_buffer_size = min(pose_circular_buffer_size + 1, pose_circular_buffer.shape[0])
                    do_skip_pose = False
                if not args.disable_time_check and time() - last_image_add_time < 0.5:
                    do_skip_pose = True
        else:
            point_references = None
            do_skip_pose = True

        if LIVE or args.force_visualization:
            text_color = (255,120,0)
            if img_avg_reproj_err is not None:
                if img_avg_reproj_err < 1:
                    text_color = (0, 255, 0)
                else:
                    text_color = (0, 0, 255)
            for img in (img_debug, board_img):
                cv2.rectangle(
                    img,
                    (0,0),
                    (180, 50),
                    (0,0,0),
                    -1
                )
                cv2.putText(
                    img,
                    f"Reproj Err: {img_avg_reproj_err:.2f}" if img_avg_reproj_err is not None else "Reproj Err: N/A",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                )
                cv2.putText(
                    img,
                    f"N good imgs: {num_total_images_used}",
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1,
                )
                cv2.putText(
                    img,
                    f"Originality: {closest_pose_dist/500:.2f}" if closest_pose_dist is not None else "Originality: N/A",
                    (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1,
                )
            img_debug = cv2.resize(img_debug, (1024, 576))
            cv2.imshow("calib", img_debug)
            cv2.imshow('undistorted', cv2.resize(undistorted_debug, (1024, 576)))
            key = cv2.waitKey(1)
            
            # Update camera properties if using v4l2 source
            if args.source == "v4l2" and 'camera_controller' in locals():
                camera_controller.update_properties()
        else:
            key = 1
        shape = img_bgr.shape[:2]
        if not do_skip_pose and img_avg_reproj_err is not None and img_avg_reproj_err > 1 and len(point_references.object_points) > 4:
            total_object_points.append(point_references.object_points)
            total_image_points.append(point_references.image_points)
            num_total_images_used +=1
            is_time_to_calib = num_total_images_used % CALIB_BATCH_SIZE == 0
            last_image_add_time = time()

            if LIVE:
                calibration_criteria_met = num_total_images_used >= CALIB_BATCH_SIZE and is_time_to_calib
            else:
                # For ReplayCamera, calibrate when we have enough images or when we're at the end
                calibration_criteria_met = (num_total_images_used >= CALIB_BATCH_SIZE and is_time_to_calib) or (index >= camera.get_total_frames())

            if calibration_criteria_met:
                sample_indices = np.random.choice(np.arange(num_total_images_used), min(60, num_total_images_used))
                run_calibration(sample_indices)
                # if num_total_images_used <= CALIB_BATCH_SIZE:
                #     # flags = None
                #     flags = cv2.CALIB_FIX_TANGENT_DIST
                # elif num_total_images_used <= 2*CALIB_BATCH_SIZE:
                #     flags = None
                # else:
                #     flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL 
                flags = None
                # For fisheye calibration, we use 4 distortion coefficients
                last_nonzero_dist_coef_limit = 4
                # Generate undistortion maps based on calibration model
                if args.calibration_model == "fisheye":
                    # OpenCV fisheye functions return different numbers of values depending on version
                    new_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_mat, dist_coeffs, DIM, None, None)
                    
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(cam_mat, dist_coeffs, None, new_cam_mat, DIM, cv2.CV_16SC2) # type: ignore
                else:
                    # For normal calibration, we don't need to pre-generate maps
                    # cv2.undistort() will handle the undistortion directly
                    pass
            if LIVE or args.force_visualization:
                cv2.imwrite(f'{imgs_path}/{len(list(imgs_path.glob("*.png")))}.png', img_bgr)

        if key == ord("q"):
            break
    
    run_calibration(np.arange(num_total_images_used))
    # Cleanup
    if not LIVE and hasattr(camera, 'close'):
        camera.close()
        print("ReplayCamera closed successfully.")
    
    # Cleanup camera control window if it exists
    if args.source == "v4l2" and 'camera_controller' in locals():
        cv2.destroyWindow("Camera Controls")
        print("Camera control window closed.")
