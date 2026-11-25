"""The slot a model plugs into, in clank's terms: arms keyed by side, cameras by role."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

from clank.types import JointState, PoseChunk


class PolicyNotReady(Exception):
    """The model cannot answer yet but may later: `reason` says why, `retry_after_s` when to ask again."""

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
        """Takes one camera tick: frames by role, arm states and tool poses by side, the prompt, capture time."""
        ...

    def infer(self) -> dict[str, PoseChunk]:
        """One chunk per side from the latest observation; PolicyNotReady when asking again later may help."""
        ...

    def reset_episode(self) -> None:
        """Forgets the observations so far: the next observe() starts a new episode."""
        ...

    def close(self) -> None:
        """Lets go of what the policy holds; later calls raise RuntimeError, and closing again does nothing."""
        ...
