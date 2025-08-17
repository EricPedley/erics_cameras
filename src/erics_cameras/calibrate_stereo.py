import cv2 as cv

from typing import NamedTuple

import numpy as np
import os

from erics_cameras import GstCamera
from time import strftime, time
from pathlib import Path
from typing import Any

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera calibration script.")
    parser.add_argument(
        "--cam_type", help="Type of camera to use for calibration.", choices=["0", "1", "2"], default=None
    )

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


    SQUARE_LENGTH = 5
    MARKER_LENGTH = 3
    NUMBER_OF_SQUARES_VERTICALLY = 11
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    charuco_marker_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    charuco_board = cv.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=charuco_marker_dictionary,
    )

    cam_mat = np.array([[297.80062345,0.,685.72493754],[0.,298.63865273,451.61133244],[0.,0.,1.,]])
    # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, Tx, Ty
    dist_coeffs = np.array([[-0.20148179,0.03270111,0.,0.,-0.00211291]])

    matrix_l = cam_mat
    matrix_r = cam_mat
    dist_l = dist_coeffs
    dist_r = dist_coeffs

    total_object_points = []
    total_image_points = []
    num_total_images_used = 0
    last_image_add_time = time()

    # Add lists for stereo calibration
    stereo_object_points = []
    stereo_image_points_l = []
    stereo_image_points_r = []

    LIVE = bool(os.getenv("LIVE", True))

    if LIVE:
        logs_base = Path("logs")
        time_dir = Path(strftime("%Y-%m-%d/%H-%M"))
        logs_path = logs_base / time_dir

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

        cap0 = cv.VideoCapture(pipeline_cam0, cv.CAP_GSTREAMER)
        cap1 = cv.VideoCapture(pipeline_cam1, cv.CAP_GSTREAMER)
        # cap0 = cv.VideoCapture('/home/dpsh/kscalecontroller/pi/left_video.avi')
        # cap1 = cv.VideoCapture('/home/dpsh/kscalecontroller/pi/right_video.avi')
        # camera.start_recording()
        # cv.namedWindow("calib", cv.WINDOW_NORMAL)
        cv.namedWindow("left", cv.WINDOW_NORMAL)
        cv.namedWindow("right", cv.WINDOW_NORMAL)
        # cv.resizeWindow("calib", (1024, 576))
        # board_img = cv.cvtColor(cv.rotate(charuco_board.generateImage((1080,1920), marginSize=10), cv.ROTATE_90_CLOCKWISE), cv.COLOR_GRAY2BGR)
        # cv.resizeWindow("charuco_board", (1600,900))

    index = 0
    imgs_path = logs_path / "calib_imgs"
    imgs_path.mkdir(exist_ok=True, parents=True)
    images = sorted(list(imgs_path.glob("*.png")))

    det_results: list[BoardDetectionResults] = []

    latest_error = None

    pose_circular_buffer = np.empty((100, 6), dtype=np.float32)
    pose_circular_buffer_index = 0
    pose_circular_buffer_size = 0

    last_detection_results = None

    while True:
        if LIVE:
            # img_l = camera_l.take_image()
            # img_r = camera_r.take_image()

            if not (cap0.grab() and cap1.grab()):
                print("❌ Failed to grab from one or both cameras")
                continue

            # Decode after pulling both from buffers
            ret_l, img_bgr_l = cap0.retrieve()
            ret_r, img_bgr_r = cap1.retrieve()

            if not ret_l or not ret_r:
                print("Failed to get image")
                continue
        else:
            if index == len(images):
                break
            img_bgr_l = cv.imread(f"{images[index]}")  # Use left image
            img_bgr_r = cv.imread(f"{images[index]}")  # Use right image (replace with correct path if needed)
            index += 1
            print(f"Processing image {index}/{len(images)}")

        img_l_debug = img_bgr_l.copy()
        img_r_debug = img_bgr_r.copy()
        # img_debug = img_bgr

        img_gray_l = cv.cvtColor(img_bgr_l, cv.COLOR_BGR2GRAY)
        img_gray_r = cv.cvtColor(img_bgr_r, cv.COLOR_BGR2GRAY)
        charuco_detector = cv.aruco.CharucoDetector(charuco_board)
        detection_results_l = BoardDetectionResults(*charuco_detector.detectBoard(img_gray_l))
        detection_results_r = BoardDetectionResults(*charuco_detector.detectBoard(img_gray_r))

        img_avg_reproj_err = None
        closest_pose_dist = 0 # TODO: refactor. This makes it really hard to keep track of where this variable is scoped
        # Find common IDs between left and right detections

        if (
            detection_results_l.charuco_corners is not None and
            detection_results_r.charuco_corners is not None and
            len(detection_results_l.charuco_corners) > 4 and
            len(detection_results_r.charuco_corners) > 4 and
            len(common_ids := set(ids_l:=detection_results_l.charuco_ids.squeeze().tolist()).intersection(set(ids_r:=detection_results_r.charuco_ids.squeeze().tolist()))) > 4
        ):
            # Get corners for common IDs
            corners_l = np.array([corner for id, corner in zip(ids_l, detection_results_l.charuco_corners) if id in common_ids])
            corners_r = np.array([corner for id, corner in zip(ids_r, detection_results_r.charuco_corners) if id in common_ids])
            ids_common = np.array(list(common_ids)).reshape((-1, 1))

            point_refs_l = PointReferences(*charuco_board.matchImagePoints(corners_l, ids_common))
            point_refs_r = PointReferences(*charuco_board.matchImagePoints(corners_r, ids_common))

            full_refs_l = PointReferences(*charuco_board.matchImagePoints(np.array(detection_results_l.charuco_corners), np.array(ids_l).reshape((-1,1))))
            full_refs_r = PointReferences(*charuco_board.matchImagePoints(np.array(detection_results_r.charuco_corners), np.array(ids_r).reshape((-1,1))))


            for pt in full_refs_l.image_points.squeeze():
                cv.circle(img_l_debug, tuple(pt.astype(int)), 7, (255,0,0), -1)
            for pt in full_refs_r.image_points.squeeze():
                cv.circle(img_r_debug, tuple(pt.astype(int)), 7, (255,0,0), -1)
            # Optionally visualize points on debug images
            for pt in point_refs_l.image_points.squeeze():
                cv.circle(img_l_debug, tuple(pt.astype(int)), 5, (0,255,0), -1)
            for pt in point_refs_r.image_points.squeeze():
                cv.circle(img_r_debug, tuple(pt.astype(int)), 5, (0,255,0), -1)


            ret, rvecs, tvecs = cv.solvePnP(
                point_refs_l.object_points,
                point_refs_r.image_points,
                cam_mat,
                dist_coeffs,
                flags=cv.SOLVEPNP_IPPE,
            )
            if ret:
                reproj: np.ndarray = cv.projectPoints(
                    point_refs_l.object_points, rvecs, tvecs, cam_mat, dist_coeffs
                )[0].squeeze()

                image_points = point_refs_l.image_points.squeeze()


                img_avg_reproj_err = np.mean(
                    np.linalg.norm(
                        image_points - reproj, axis=1
                    )
                )
                
                movement_magnitude=1e9
                if last_detection_results is not None:
                    current_ids = [id for id in detection_results_l.charuco_ids.squeeze().tolist()]
                    last_ids = [id for id in last_detection_results.charuco_ids.squeeze().tolist()]
                    intersecting_ids = [i for i in current_ids if i in last_ids]
                    if len(intersecting_ids) > 2:
                        current_intersect_charuco_corners = np.array([
                            corner
                            for id, corner in zip(
                                current_ids,
                                detection_results_l.charuco_corners
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
                last_detection_results = detection_results_l

                for pt in image_points:
                    green_amount = int((1-np.tanh(4*(movement_magnitude-1.5)))/4 *255) if movement_magnitude>1 else 255
                    cv.circle(
                        img_bgr_l, tuple(pt.astype(int)), 7, (255, green_amount, 0), -1
                    )
                for pt in reproj:
                    if np.any(np.isnan(pt)) or np.any(pt<0):
                        continue
                    try:
                        cv.circle(img_bgr_l, tuple(pt.astype(int)), 5,(0, 0, 255) if img_avg_reproj_err > 1 else (0,255,0),-1)
                    except:
                        print("Error in cv circle")


                
            if rvecs is None or tvecs is None :
                do_skip_pose = True
            else:
                combo_vec = np.concatenate((rvecs.squeeze(), tvecs.squeeze()))
                pose_too_close = pose_circular_buffer_size > 0 and (closest_pose_dist:=np.min(np.linalg.norm(pose_circular_buffer[:pose_circular_buffer_size] - combo_vec.reshape((1,6)), axis=1))) < 500
                if pose_too_close or movement_magnitude>1:
                    do_skip_pose = True
                else:
                    pose_circular_buffer[pose_circular_buffer_index] = combo_vec
                    pose_circular_buffer_index = (pose_circular_buffer_index + 1) % pose_circular_buffer.shape[0]
                    pose_circular_buffer_size = min(pose_circular_buffer_size + 1, pose_circular_buffer.shape[0])
                    do_skip_pose = False
                if time() - last_image_add_time < 0.5:
                    do_skip_pose = True
        else:
            point_refs_l = None
            do_skip_pose = True

        if LIVE:
            text_color = (255,120,0)
            if img_avg_reproj_err is not None:
                if img_avg_reproj_err < 1:
                    text_color = (0, 255, 0)
                else:
                    text_color = (0, 0, 255)
            for img in (img_l_debug, img_r_debug):
                cv.rectangle(
                    img,
                    (0,0),
                    (180, 50),
                    (0,0,0),
                    -1
                )
                cv.putText(
                    img,
                    f"Reproj Err: {img_avg_reproj_err:.2f}" if img_avg_reproj_err is not None else "Reproj Err: N/A",
                    (5, 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                )
                cv.putText(
                    img,
                    f"N good imgs: {num_total_images_used}",
                    (5, 25),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1,
                )
                cv.putText(
                    img,
                    f"Originality: {closest_pose_dist/500:.2f}" if closest_pose_dist is not None else "Originality: N/A",
                    (5, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1,
                )
            cv.imshow("left", cv.resize(img_l_debug, (1024, 576)))
            cv.imshow("right", cv.resize(img_r_debug, (1024, 576)))
            key = cv.waitKey(1)
        else:
            key = 1
        shape = img_bgr_l.shape[:2]
        if not do_skip_pose and img_avg_reproj_err is not None and img_avg_reproj_err > 1 and point_refs_l is not None and len(point_refs_l.object_points) > 4:
            total_object_points.append(point_refs_l.object_points)
            total_image_points.append(point_refs_l.image_points)
            CALIB_BATCH_SIZE = 15
            num_total_images_used +=1

            # Add to stereo lists
            stereo_object_points.append(point_refs_l.object_points)
            stereo_image_points_l.append(point_refs_l.image_points)
            stereo_image_points_r.append(point_refs_r.image_points)
            is_time_to_calib = num_total_images_used % CALIB_BATCH_SIZE == 0
            last_image_add_time = time()

            if LIVE:
                calibration_criteria_met = num_total_images_used >= CALIB_BATCH_SIZE and is_time_to_calib
            else:
                calibration_criteria_met = index == len(images)

            if calibration_criteria_met:
                sample_indices = np.random.choice(np.arange(num_total_images_used), min(60, num_total_images_used))
                if num_total_images_used <= CALIB_BATCH_SIZE:
                    flags = None
                elif num_total_images_used <= 2*CALIB_BATCH_SIZE:
                    flags = cv.CALIB_RATIONAL_MODEL
                else:
                    flags = cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL 
                flags = None
                last_nonzero_dist_coef_limit = max([5]+[i+1 for i in range(5,len(dist_coeffs)) if dist_coeffs[0,i]!=0.0])
                ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(stereo_object_points, stereo_image_points_l, stereo_image_points_r, matrix_l, dist_l, matrix_r, dist_r, (1280, 960))

                print(R,T)
                # print(f'Reproj error: {calibration_results.repError}')
                # latest_error = calibration_results.repError
                # matrix_outputs = '\n'.join([
                #     f'cam_mat = np.array({",".join(str(calibration_results.camMatrix).split())})'.replace('[,','['),
                #     '# k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, Tx, Ty',
                #     f'dist_coeffs = np.array({",".join(str(calibration_results.distcoeff).split())})'.replace('[,','[')
                # ])
                
                num_batches = num_total_images_used//CALIB_BATCH_SIZE
                calib_path = logs_path / 'intrinsics'
                calib_path.mkdir(exist_ok=True, parents=True)

                with open(f'{calib_path}/{num_batches}.txt', 'w+') as f:
                    f.write(f'{R},{T}')
            if LIVE:
                cv.imwrite(f'{imgs_path}/{len(list(imgs_path.glob("*.png")))}.png', img_bgr_l)

        # Run stereo calibration when enough pairs collected
        STEREO_CALIB_BATCH_SIZE = 15
        if len(stereo_object_points) >= STEREO_CALIB_BATCH_SIZE:
            print("Running stereo calibration...")
            ret, cam_mat_l, dist_l, cam_mat_r, dist_r, R, T, E, F = cv.stereoCalibrate(
                stereo_object_points,
                stereo_image_points_l,
                stereo_image_points_r,
                cam_mat, dist_coeffs,
                cam_mat, dist_coeffs,
                img_bgr_l.shape[:2][::-1],
                flags=cv.CALIB_FIX_INTRINSIC
            )
            print(f"Stereo reprojection error: {ret}")
            print("Left Camera Matrix:\n", cam_mat_l)
            print("Right Camera Matrix:\n", cam_mat_r)
            print("Rotation:\n", R)
            print("Translation:\n", T)
            # Save results if needed
            # Reset lists to avoid repeated calibration
            stereo_object_points.clear()
            stereo_image_points_l.clear()
            stereo_image_points_r.clear()

        if key == ord("q"):
            break
