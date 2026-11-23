"""The data types every clank stage agrees on."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PoseChunk:
    """A planned tool path for one arm, in that arm's own base frame.

    t: (N,) seconds on the client's monotonic clock, increasing.
    T_base_tcp: (N, 4, 4) pose of the tool centre point at each time.
    grip: (N,) gripper aperture, 0 closed to 1 open.
    """

    t: npt.NDArray[np.float64]
    T_base_tcp: npt.NDArray[np.float64]
    grip: npt.NDArray[np.float64]


@dataclass(frozen=True)
class JointState:
    """One arm's measured state.

    t: when it was read, seconds on the client's monotonic clock.
    q: (n,) joint positions, radians, in the URDF's joint order.
    grip: gripper aperture, 0 closed to 1 open.
    """

    t: float
    q: npt.NDArray[np.float64]
    grip: float
