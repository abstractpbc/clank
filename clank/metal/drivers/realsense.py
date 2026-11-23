"""RealSense cameras, through librealsense."""

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs


class RealSenseCamera:
    """The colour stream of one RealSense, picked by its librealsense serial."""

    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self._serial = serial
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

    def read(self) -> npt.NDArray[np.uint8]:
        """The next frame, (H, W, 3) BGR; waits for it."""
        color = self._pipeline.wait_for_frames().get_color_frame()
        if not color:
            raise RuntimeError(f"RealSense {self._serial} returned a frame set with no colour frame")
        return np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        self._pipeline.stop()
