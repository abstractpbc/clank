"""Tool poses as the protocols send and receive them: rigid (4, 4) transforms, and rotation6d rows.

rotation6d is a rotation matrix's first column followed by its second; the third is their cross product.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

## How close to zero a rotation6d column's length may come before it names no direction.
_DEGENERATE = 1e-6
## How far a rigid transform may stray from orthonormal, from a bottom row of (0, 0, 0, 1), and from a determinant of 1.
_RIGID_ATOL = 1e-6


def rigid(value: Any, name: str) -> npt.NDArray[np.float64]:
    """`value` as a (4, 4) float64 rigid transform with a proper rotation; ValueError naming `name` otherwise."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    pose = np.asarray(raw, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be finite with shape (4, 4), got {pose.shape}")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], (0, 0, 0, 1), atol=_RIGID_ATOL, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_RIGID_ATOL, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=_RIGID_ATOL, rtol=0)
    ):
        raise ValueError(f"{name} must be a rigid transform with a proper rotation")
    return pose


def rot6d(T: npt.NDArray[np.float64]) -> list[float]:
    """A pose's rotation as rotation6d: its matrix's first column, then its second."""
    return [*T[:3, 0], *T[:3, 1]]


def from_rot6d(rows: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    """(N, 9) rows of xyz and rotation6d as (N, 4, 4) poses: the two columns made orthonormal (Gram-Schmidt), the
    third their cross product. ValueError naming `name` for a column of no length or two parallel columns."""
    first, second = rows[:, 3:6], rows[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if not np.isfinite(first_norm).all() or np.any(first_norm < _DEGENERATE):
        raise ValueError(f"{name}: a rotation6d has a zero first column")
    b1 = first / first_norm
    b2 = second - np.sum(b1 * second, axis=1, keepdims=True) * b1
    second_norm = np.linalg.norm(b2, axis=1, keepdims=True)
    if not np.isfinite(second_norm).all() or np.any(second_norm < _DEGENERATE):
        raise ValueError(f"{name}: a rotation6d has parallel columns")
    b2 = b2 / second_norm
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
    poses[:, :3, 3] = rows[:, :3]
    return poses
