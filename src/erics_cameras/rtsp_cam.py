from .gst_cam import GstCamera
from pathlib import Path


class RTSPCamera(GstCamera):

    def __init__(
        self,
        log_dir: str | Path | None = None,
        url_str: str = "192.168.42.1/h264",
        flip=False,
    ):
        """
        TODO: make resolution choosable. Hard-coded in JetsonCamera rn.
        """
        flip_str = '! videoflip method=rotate-180' if flip else ''
        pipeline = f'rtspsrc location=rtsp://{url_str} latency=0 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! videoconvert {flip_str} ! video/x-raw, format=BGRx ! appsink drop=true'
        super().__init__(log_dir, pipeline)