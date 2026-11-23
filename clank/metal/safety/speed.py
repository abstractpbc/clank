"""How fast an arm's joints may be told to move."""

import numpy as np
import numpy.typing as npt


def limit_step(
    previous: npt.NDArray[np.float64], wanted: npt.NDArray[np.float64], dt: float, limit: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], bool]:
    """The joints after `previous` on the way to `wanted`: the whole step, shortened where needed so that no joint
    goes further than its `limit` (rad/s) allows in `dt` seconds; and whether it was shortened. The step is shortened
    as a whole, not joint by joint, so it keeps its direction and the tool still heads where it was going."""
    step = wanted - previous
    over = float(np.max(np.abs(step) / (limit * dt)))
    if over <= 1.0:
        return wanted, False
    return previous + step / over, True


def travel_s(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64], limit: npt.NDArray[np.float64]) -> float:
    """How long a straight line in joint space from `start` to `end` takes with no joint over its `limit` (rad/s)."""
    return float(np.max(np.abs(end - start) / limit))
