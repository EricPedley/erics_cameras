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


    SQUARE_LENGTH = 500
    MARKER_LENGHT = 300
    NUMBER_OF_SQUARES_VERTICALLY = 11
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    charuco_marker_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    charuco_board = cv.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGHT,
        dictionary=charuco_marker_dictionary,
    )

    cam_mat = np.array([[1000, 0, 1920 / 2], [0, 1000, 1080 / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((1,5), dtype=np.float32)

    total_object_points = []
    total_image_points = []
    num_total_images_used = 0
    last_image_add_time = time()

    LIVE = bool(os.getenv("LIVE", True))

    if LIVE:
        logs_base = Path("logs")
        time_dir = Path(strftime("%Y-%m-%d/%H-%M"))
        logs_path = logs_base / time_dir

        pipeline = (
            "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36  ! "
            "video/x-raw,format=BGR,width=1280,height=960,framerate=30/1 ! "
            "videoconvert ! appsink drop=1 max-buffers=1"
        )
        camera = GstCamera("./testimages", pipeline)
        # camera.start_recording()
        # cv.namedWindow("calib", cv.WINDOW_NORMAL)
        cv.namedWindow("charuco_board", cv.WINDOW_NORMAL)
        # cv.resizeWindow("calib", (1024, 576))
        board_img = cv.cvtColor(cv.rotate(charuco_board.generateImage((1080,1920), marginSize=10), cv.ROTATE_90_CLOCKWISE), cv.COLOR_GRAY2BGR)
        cv.resizeWindow("charuco_board", (1600,900))

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
            img = camera.take_image()
            if img is None:
                print("Failed to get image")
                continue
            img_bgr = img.get_array()
            # cv.imshow("debug", img_bgr)
        else:
            if index == len(images):
                break
            img_bgr = cv.imread(f"{images[index]}")
            index += 1
            print(f"Processing image {index}/{len(images)}")

        img_debug = img_bgr.copy()
        # img_debug = img_bgr

        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        charuco_detector = cv.aruco.CharucoDetector(charuco_board)
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

            ret, rvecs, tvecs = cv.solvePnP(
                point_references.object_points,
                point_references.image_points,
                cam_mat,
                dist_coeffs,
                flags=cv.SOLVEPNP_IPPE,
            )
            if ret:
                reproj: np.ndarray = cv.projectPoints(
                    point_references.object_points, rvecs, tvecs, cam_mat, dist_coeffs
                )[0].squeeze()

                image_points = point_references.image_points.squeeze()


                img_avg_reproj_err = np.mean(
                    np.linalg.norm(
                        image_points - reproj, axis=1
                    )
                )
                
                movement_magnitude=1e9
                if last_detection_results is not None:
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

                for pt in image_points:
                    green_amount = int((1-np.tanh(4*(movement_magnitude-1.5)))/4 *255) if movement_magnitude>1 else 255
                    cv.circle(
                        img_debug, tuple(pt.astype(int)), 7, (255, green_amount, 0), -1
                    )
                for pt in reproj:
                    if np.any(np.isnan(pt)) or np.any(pt<0):
                        continue
                    try:
                        cv.circle(img_debug, tuple(pt.astype(int)), 5,(0, 0, 255) if img_avg_reproj_err > 1 else (0,255,0),-1)
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
            point_references = None
            do_skip_pose = True

        if LIVE:
            text_color = (255,120,0)
            if img_avg_reproj_err is not None:
                if img_avg_reproj_err < 1:
                    text_color = (0, 255, 0)
                else:
                    text_color = (0, 0, 255)
            for img in (img_debug, board_img):
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
            img_debug = cv.resize(img_debug, (1024, 576))
            cv.imshow("calib", img_debug)
            key = cv.waitKey(1)
        else:
            key = 1
        shape = img_bgr.shape[:2]
        if not do_skip_pose and img_avg_reproj_err is not None and img_avg_reproj_err > 1 and len(point_references.object_points) > 4:
            total_object_points.append(point_references.object_points)
            total_image_points.append(point_references.image_points)
            CALIB_BATCH_SIZE = 15
            num_total_images_used +=1
            is_time_to_calib = num_total_images_used % CALIB_BATCH_SIZE == 0
            last_image_add_time = time()

            if LIVE:
                calibration_criteria_met = num_total_images_used >= CALIB_BATCH_SIZE and is_time_to_calib
            else:
                calibration_criteria_met = index == len(images)

            if calibration_criteria_met:
                sample_indices = np.random.choice(np.arange(num_total_images_used), min(60, num_total_images_used))
                # if num_total_images_used <= CALIB_BATCH_SIZE:
                #     flags = None
                # elif num_total_images_used <= 2*CALIB_BATCH_SIZE:
                #     flags = cv.CALIB_RATIONAL_MODEL
                # else:
                #     flags = cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL 
                flags = cv.CALIB_FIX_S1_S2_S3_S4
                last_nonzero_dist_coef_limit = max([5]+[i+1 for i in range(5,len(dist_coeffs)) if dist_coeffs[0,i]!=0.0])
                calibration_results = CameraCalibrationResults(
                    *cv.calibrateCamera(
                        [total_object_points[i] for i in sample_indices],
                        [total_image_points[i] for i in sample_indices],
                        shape,
                        None if flags is None else cam_mat,  # type: ignore
                        None if flags is None else dist_coeffs[:,:last_nonzero_dist_coef_limit],  # type: ignore
                        flags=flags
                    )
                )

                print(f'Reproj error: {calibration_results.repError}')
                latest_error = calibration_results.repError
                matrix_outputs = '\n'.join([
                    f'cam_mat = np.array({",".join(str(calibration_results.camMatrix).split())})'.replace('[,','['),
                    '# k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, Tx, Ty',
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
                cv.imwrite(f'{imgs_path}/{len(list(imgs_path.glob("*.png")))}.png', img_bgr)

        if key == ord("q"):
            break
