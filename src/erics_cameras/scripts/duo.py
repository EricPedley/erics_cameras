import cv2
import numpy as np

cam_mat = np.array([[266.61728276,0.,643.83126137],[0.,266.94450686,494.81811813],[0.,0.,1.,]], dtype=np.float32)
DIM = (1280, 960)

dist_coeffs = np.array([[-6.07417419e-02,9.95447444e-02,-2.26448001e-04,1.22881804e-03,3.42134205e-03,1.45361886e-01,8.03248099e-02,2.11170107e-02,-3.80620047e-03,2.48350591e-05,-8.33565666e-04,2.97806723e-05]])
DATA=f"<?xml version=\"1.0\"?><opencv_storage><cameraMatrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>f</dt><data>2.85762378e+03 0. 1.93922961e+03 0. 2.84566113e+03 1.12195850e+03 0. 0. 1.</data></cameraMatrix><distCoeffs type_id=\"opencv-matrix\"><rows>5</rows><cols>1</cols><dt>f</dt><data>-6.14039421e-01 4.00045455e-01 1.47132971e-03 2.46772077e-04 -1.20407566e-01</data></distCoeffs></opencv_storage>"


# Pipeline for camera 0
pipeline_cam0 = (
    "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36 ! "
    "video/x-raw,format=BGR,width=1280,height=960,framerate=45/1 ! "
    f"videoconvert !appsink drop=1 max-buffers=1"
)

# Pipeline for camera 1
pipeline_cam1 = (
    "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36 ! "
    "video/x-raw,format=BGR,width=1280,height=960,framerate=45/1 ! "
    f"videoconvert ! appsink drop=1 max-buffers=1"
)

cap0 = cv2.VideoCapture(pipeline_cam0, cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(pipeline_cam1, cv2.CAP_GSTREAMER)

width, height, fps = 1280, 960, 45

gst_str = (
    f'appsrc ! videoconvert ! video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 '
    '! shmsink socket-path=/tmp/video_pipe sync=false wait-for-connection=false shm-size=10000000'
)

# read with gst-launch-1.0 shmsrc socket-path=/tmp/video_pipe do-timestamp=true ! video/x-raw,format=BGR,width=1280,height=960,framerate=45/1 ! videoconvert ! autovideosink


writer = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)


if not cap0.isOpened() or not cap1.isOpened():
    print("❌ Failed to open one or both cameras")
    exit()

while True:
    # Grab frames from both cameras as close together as possible
    if not (cap0.grab() and cap1.grab()):
        print("❌ Failed to grab from one or both cameras")
        break

    # Retrieve frames after grabbing
    ret0, frame0 = cap0.retrieve()
    ret1, frame1 = cap1.retrieve()

    frame0 = cv2.undistort(frame0, cam_mat, dist_coeffs)
    frame1 = cv2.undistort(frame1, cam_mat, dist_coeffs)

    writer.write(frame0)

    if not ret0 or not ret1:
        print("❌ Failed to retrieve from one or both cameras")
        break

    # Concatenate images side-by-side
    combined = cv2.hconcat([frame0, frame1])

    cv2.imshow("Both Cameras Side-by-Side", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
writer.release()
cv2.destroyAllWindows()
