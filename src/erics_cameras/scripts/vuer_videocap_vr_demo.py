import cv2
from vuer import Vuer, VuerSession
import asyncio
from vuer.schemas import ImageBackground

async def stream_cameras(session: VuerSession, left_src=0, right_src=1):
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
    while True:
        ret_left, frame_left = cap1.read()
        ret_right, frame_right = cap2.read()
        if not ret_left or not ret_right:
            continue
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_left_rgb, "Left Camera", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(frame_right_rgb, "Right Camera", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        # Send both images as ImageBackground objects for left/right eye
        session.upsert([
            ImageBackground(
                frame_left_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=1,
                format="jpeg",
                quality=50,
                key="background-left",
                interpolate=True,
            ),
            ImageBackground(
                frame_right_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=2,
                format="jpeg",
                quality=50,
                key="background-right",
                interpolate=True,
            ),
        ], to="bgChildren")
        await asyncio.sleep(1/30)  # ~30 FPS for smoother streaming

if __name__ == "__main__":
    app = Vuer()
    @app.spawn(start=True)
    async def main(session: VuerSession):
        await stream_cameras(session)
    app.run()
