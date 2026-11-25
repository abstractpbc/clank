"""What a workcell declares for each arm's safety: its joints' speed limits and its tool's workspace."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from clank.metal.safety import speed
from clank.metal.safety.workspace import Box


@dataclass(frozen=True)
class Envelope:
    """Each joint's speed limit, rad/s, below its top speed; and the tool's workspace box, or None."""

    speed_limit: tuple[float, ...]
    workspace: Box | None

    def __post_init__(self) -> None:
        limit = np.asarray(self.speed_limit, dtype=np.float64)
        if limit.ndim != 1 or len(limit) == 0 or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError(f"speed_limit is one finite, positive speed per joint, not {self.speed_limit}")

    def require_inside(self, T_base_tcp: npt.NDArray[np.float64], what: str, slack_m: float = 0.0) -> None:
        """Raises ValueError when the tool is more than `slack_m` outside the workspace."""
        if self.workspace is None:
            return
        position = T_base_tcp[:3, 3]
        outside = float(np.max(np.maximum(self.workspace.low - position, position - self.workspace.high)))
        if outside > slack_m:
            x, y, z = position
            raise ValueError(
                f"{what} is at x={x:.3f} y={y:.3f} z={z:.3f} m, {outside * 1000:.0f} mm outside its workspace "
                f"{self.workspace.low} to {self.workspace.high}"
            )

    def inside(self, T_base_tcp: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], bool]:
        """The tool target moved into the workspace; and whether it had to move."""
        if self.workspace is None:
            return T_base_tcp, False
        return self.workspace.clamp(T_base_tcp)

    def step(
        self, previous: npt.NDArray[np.float64], wanted: npt.NDArray[np.float64], dt: float
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """The joints `dt` after `previous` toward `wanted` within the limits, and whether the step was shortened."""
        return speed.limit_step(previous, wanted, dt, np.asarray(self.speed_limit))

    def travel_s(self, start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]) -> float:
        """How long a straight line in joint space from `start` to `end` takes within the speed limits."""
        return speed.travel_s(start, end, np.asarray(self.speed_limit))
