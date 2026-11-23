"""The slot a model plugs into: fed every camera tick, asked for a chunk when the trajectory needs one.

Everything here is in clank's own terms, so the loop that drives a robot names no provider. A protocol turns these
into what its service speaks: arms are keyed by side (`left`, `right`) and cameras by role (`overhead`, `wrist_left`,
`wrist_right`).
"""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

from clank.types import JointState, PoseChunk


class PolicyNotReady(Exception):
    """The model cannot answer yet, and asking again later may work: it is starting, busy, or took too long.

    reason: why, in the service's words.
    retry_after_s: how long to wait before asking again.
    """

    def __init__(self, reason: str, retry_after_s: float) -> None:
        super().__init__(f"{reason}; ask again in {retry_after_s:g} s")
        self.reason = reason
        self.retry_after_s = retry_after_s


class Policy(Protocol):
    def observe(
        self,
        frames: Mapping[str, npt.NDArray[np.uint8]],
        states: Mapping[str, JointState],
        poses: Mapping[str, npt.NDArray[np.float64]],
        prompt: str,
        t: float,
    ) -> None:
        """Takes one camera tick.

        frames: camera role -> (H, W, 3) BGR image.
        states: side -> that arm's measured state.
        poses: side -> where that arm's tool is, (4, 4) in the arm's own base frame: its joints through the workcell's
            kinematics.
        prompt: the task, in words.
        t: when the frames were captured, seconds on the client's monotonic clock.
        """
        ...

    def infer(self) -> dict[str, PoseChunk]:
        """Returns one chunk per arm, keyed by side, planned from the latest observation.

        Raises PolicyNotReady when the model cannot answer yet. Any other error means asking again will not help.
        """
        ...

    def reset_episode(self) -> None:
        """Forgets the observations so far: the next observe() starts a new episode."""
        ...

    def close(self) -> None:
        """Lets go of what the policy holds. observe(), infer() and reset_episode() raise RuntimeError from then on;
        closing again does nothing."""
        ...
