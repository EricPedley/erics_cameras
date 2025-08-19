import cv2
from vuer import Vuer, VuerSession
import asyncio
from vuer.schemas import ImageBackground
from erics_cameras import RTSPCamera

async def stream_cameras(session: VuerSession, left_src=0, right_src=1):
    left_cap = RTSPCamera()
    while True:
        img_left = left_cap.take_image()
        img_right = left_cap.take_image()
        if img_left is None or img_right is None:
            continue
        
        frame_left = img_left.get_array()
        frame_right = img_right.get_array()
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
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
