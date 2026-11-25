"""The data types every clank stage agrees on."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PoseChunk:
    """One arm's planned tool path in its base frame: times (N,), poses (N, 4, 4), apertures (N,)."""

    t: npt.NDArray[np.float64]
    T_base_tcp: npt.NDArray[np.float64]
    grip: npt.NDArray[np.float64]


@dataclass(frozen=True)
class JointState:
    """One arm's measured state: read time, joint positions (n,) in URDF order, and gripper aperture."""

    t: float
    q: npt.NDArray[np.float64]
    grip: float
