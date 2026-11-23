"""The slot one arm's driver plugs into."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from clank.types import JointState


class Robot(Protocol):
    def read(self) -> JointState:
        """The arm's state now."""
        ...

    def command(self, q: npt.NDArray[np.float64], grip: float) -> None:
        """Sends joint targets (radians, the URDF's joint order) and a gripper aperture (0 closed to 1 open)."""
        ...

    def close(self) -> None:
        """Releases the device. What the arm does then is the driver's to say: one without brakes goes limp."""
        ...
