"""
This does the same thing as `gazebo/camera_read.py` but wraps it in a different interface to be the same as `arducam.py`.
"""

from erics_cameras.camera import Camera, ImageMetadata, Image
from pathlib import Path
from time import time
import subprocess
import json

import numpy as np

import rclpy
import rclpy.node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ROSImage
from scipy.spatial.transform import Rotation
import threading
from rclpy.executors import SingleThreadedExecutor

def ensure_ros(func):
    """
    Decorator for a function to start rclpy if it has not been started yet.
    """

    def wrapper(*args, **kwargs):
        if not rclpy.ok():
            rclpy.init()

        return func(*args, **kwargs)

    return wrapper

class GazeboCameraStream(rclpy.node.Node):
    @ensure_ros
    def __init__(self, img_topic="/camera/image"):
        super().__init__("uavf_2025_gazebo_camera_stream")
        self.subscription = self.create_subscription(
            ROSImage, img_topic, self.listener_callback, 10
        )
        self.bridge = CvBridge()
        self.latest_msg: ROSImage = None
        self.custom_executor = SingleThreadedExecutor()
        self.custom_executor.add_node(self)
        self.bad_count = 0
        self._connected = False

    def listener_callback(self, msg: ROSImage):
        self.latest_msg = msg

    def connect(self):
        if self._connected:
            return False
        thread = threading.Thread(target=self.spin_thread, daemon=True)
        thread.start()
        self._connected = True
        return True

    def spin_thread(self):
        self.custom_executor.spin()

    def disconnect(self) -> None:
        self.destroy_node()

    def get_frame(self) -> np.ndarray | None:
        if self.latest_msg is None:
            self.bad_count += 1
            return None
        self.bad_count = 0
        return self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding="bgr8")


class GazeboCam(Camera):
    def __init__(
        self,
        log_dir: str | Path | None = None,
        img_topic: str = "/world/map/model/iris/model/iris_with_standoffs/link/base_link/sensor/camera/image",
    ):
        super().__init__(log_dir)
        self.stream = GazeboCameraStream(img_topic)
        self.stream.connect()
        self._topic_name = img_topic
        self._focal_len_px = 457.6 #hard-coded rn. Set to None to auto figure out but its blocking and hangs sometimes

    def take_image(self):
        img_arr = self.stream.get_frame()
        if img_arr is None:
            return None
        return Image(img_arr)

    def get_metadata(self) -> ImageMetadata:
        return ImageMetadata(
            timestamp=time(),
        )

    def get_focal_length_px(self) -> float:
        if self._focal_len_px is None:
            command = [
                "gz",
                "topic",
                "-e",
                "--json-output",
                "-t",
                self._topic_name.replace("image", "camera_info"),
                "-n",
                "1",
            ]

            # Execute the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True)

            # Check if the command was successful
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")

            # Parse the JSON output into a Python dictionary
            try:
                data_dict = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON: {e}")
            self._focal_len_px = data_dict["intrinsics"]["k"][0]
            print(f'focal length: {self._focal_len_px}')

        return self._focal_len_px
