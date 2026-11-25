"""Where an arm's tool may go: an axis-aligned box in the arm's own base frame."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Box:
    """low, high: the box's corners (x, y, z), metres in the arm's base frame; every `low` below its `high`."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def __post_init__(self) -> None:
        low, high = np.asarray(self.low, dtype=np.float64), np.asarray(self.high, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError(f"a box's corners are three finite numbers each, not {self.low} and {self.high}")
        if np.any(low >= high):
            raise ValueError(f"a box's low corner must be below its high corner on every axis: {self.low}, {self.high}")

    def contains(self, T_base_tcp: npt.NDArray[np.float64]) -> bool:
        """Whether the tool's position is inside the box, its faces included."""
        position = T_base_tcp[:3, 3]
        return bool(np.all(position >= self.low) and np.all(position <= self.high))

    def clamp(self, T_base_tcp: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], bool]:
        """The pose with its position clamped into the box, rotation kept, and whether it moved."""
        position = T_base_tcp[:3, 3]
        inside = np.clip(position, self.low, self.high)
        if np.array_equal(inside, position):
            return T_base_tcp, False
        clamped = T_base_tcp.copy()
        clamped[:3, 3] = inside
        return clamped, True
