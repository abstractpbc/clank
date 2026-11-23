"""The current plan for one arm: chunks go in as they arrive, the control loop samples it by time.

A new chunk never replaces the plan in one step. From the moment it arrives it is cross-faded in from the plan in
force over a short window, with a sigmoid weight, so the commanded pose is continuous across the seam however far the
two plans disagree. The weight is a function of time since the seam, because the plan is indexed by time and holds
tool poses rather than joint rows, and it is rescaled to run from exactly 1 (all old plan) at the seam to exactly 0
at the end of the window, so there is no step at either end.

Each chunk is smoothed before it is blended in: a Savitzky-Golay fit over its rows takes out the row-to-row wobble a
model's poses carry, which straight lines between the rows would otherwise turn into a velocity step at every row. The
fit is centred, so it adds no lag, and the gripper apertures are left as they are so that a grasp stays crisp.
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from clank.types import PoseChunk

## How sharply the cross-fade's weight turns over mid-window: `1 / (1 + exp(3 * (x - 0.5)))`.
_SIGMOID_STEEPNESS = 3.0
## The smoothing fit: a quadratic over this long a window of rows, which passes motion below about 3 Hz.
_SMOOTH_S = 0.35
_SMOOTH_ORDER = 2


class Trajectory:
    def __init__(self, blend_s: float) -> None:
        """blend_s: how long a new chunk takes to fade in from the plan in force, seconds."""
        if not blend_s > 0:
            raise ValueError(f"blend_s must be positive, got {blend_s}")
        self._blend_s = blend_s
        self._plan: PoseChunk | None = None

    def has_plan(self) -> bool:
        return self._plan is not None

    def hold(self, T_base_tcp: npt.NDArray[np.float64], grip: float, now: float) -> None:
        """Makes the plan "stay at this pose" for one blend window from `now`: what an arm at a standstill is
        doing, so that the first chunk fades in from it like any other."""
        self._plan = PoseChunk(
            t=np.array([now, now + self._blend_s]),
            T_base_tcp=np.stack([T_base_tcp, T_base_tcp]),
            grip=np.array([grip, grip], dtype=np.float64),
        )

    def update(self, chunk: PoseChunk, now: float) -> None:
        """Brings `chunk` into force from `now` on; its rows already in the past are dropped.

        With a plan in force, the new plan starts at `now` exactly where the old one is, and each row of the chunk
        inside the blend window is the old plan's pose at that time moved part of the way to the row: all old at
        `now`, all new by the end of the window. The window ends early where the old plan does. A plan in force that
        has already run out is held at its last pose (`sample_held`), so the chunk fades in from that pose over the
        whole window. With no plan in force the chunk is taken as it is, from the row before `now`.
        """
        chunk = _smoothed(chunk)
        ahead = int(np.searchsorted(chunk.t, now, side="right"))  # the first row strictly after now
        if ahead == len(chunk.t):
            raise ValueError(f"the chunk ends at {chunk.t[-1]:.3f} s, before now ({now:.3f} s)")
        old = self._plan
        if old is None:
            first = max(ahead - 1, 0)
            self._plan = PoseChunk(t=chunk.t[first:], T_base_tcp=chunk.T_base_tcp[first:], grip=chunk.grip[first:])
            return
        if now < old.t[0]:
            raise ValueError(f"now ({now:.3f} s) is before the plan in force, which starts at {old.t[0]:.3f} s")
        if now > old.t[-1]:
            # The plan ran out and the arm is held at its last pose: that pose is the plan in force, through the window.
            old = PoseChunk(
                t=np.append(old.t, now + self._blend_s),
                T_base_tcp=np.concatenate([old.T_base_tcp, old.T_base_tcp[-1:]]),
                grip=np.append(old.grip, old.grip[-1]),
            )
        seam_pose, seam_grip = _sample(old, now)
        window = min(self._blend_s, old.t[-1] - now)
        t, poses, grip = chunk.t[ahead:], chunk.T_base_tcp[ahead:].copy(), chunk.grip[ahead:].copy()
        for i, row_t in enumerate(t):
            if row_t >= now + window:
                break
            new_share = 1.0 - _old_weight((row_t - now) / window)
            old_pose, old_grip = _sample(old, row_t)
            poses[i] = _between(old_pose, poses[i], new_share)
            grip[i] = (1 - new_share) * old_grip + new_share * grip[i]
        kept = old.t < now
        self._plan = PoseChunk(
            t=np.concatenate([old.t[kept], [now], t]),
            T_base_tcp=np.concatenate([old.T_base_tcp[kept], seam_pose[None], poses]),
            grip=np.concatenate([old.grip[kept], [seam_grip], grip]),
        )

    def sample(self, t: float) -> tuple[npt.NDArray[np.float64], float]:
        """The planned tool pose (4, 4) and gripper aperture at time `t`: positions and apertures interpolated
        linearly, rotations along the shortest arc."""
        plan = self._plan
        if plan is None:
            raise RuntimeError("the trajectory holds no plan yet")
        if not plan.t[0] <= t <= plan.t[-1]:
            raise ValueError(f"t = {t:.3f} s is outside the plan, {plan.t[0]:.3f} s to {plan.t[-1]:.3f} s")
        return _sample(plan, t)

    def sample_held(self, t: float) -> tuple[npt.NDArray[np.float64], float, float]:
        """As sample(), except that past the plan's end it holds the plan's last pose and aperture.

        Returns the pose, the aperture, and how long the plan has been over at `t`: 0 while it is not. The plan is
        read once, so a chunk brought into force on another thread meanwhile cannot split the answer.
        """
        plan = self._plan
        if plan is None:
            raise RuntimeError("the trajectory holds no plan yet")
        if t < plan.t[0]:
            raise ValueError(f"t = {t:.3f} s is before the plan, which starts at {plan.t[0]:.3f} s")
        if t > plan.t[-1]:
            return plan.T_base_tcp[-1].copy(), float(plan.grip[-1]), t - float(plan.t[-1])
        pose, grip = _sample(plan, t)
        return pose, grip, 0.0


def _smoothed(chunk: PoseChunk) -> PoseChunk:
    """`chunk` with its positions and rotations smoothed by a Savitzky-Golay fit over `_SMOOTH_S` of rows. Rotations
    are fitted as rotation vectors from the chunk's first row. A chunk with too few rows to fit is taken as it is."""
    row_s = float(np.median(np.diff(chunk.t)))
    rows = len(chunk.t)
    # The fit's window has an odd number of rows, and no more than the chunk has.
    window = min(int(round(_SMOOTH_S / row_s)) | 1, rows if rows % 2 else rows - 1)
    if window <= _SMOOTH_ORDER + 1:
        return chunk
    poses = chunk.T_base_tcp.copy()
    poses[:, :3, 3] = savgol_filter(chunk.T_base_tcp[:, :3, 3], window, _SMOOTH_ORDER, axis=0, mode="interp")
    first = Rotation.from_matrix(chunk.T_base_tcp[0, :3, :3])
    turns = (first.inv() * Rotation.from_matrix(chunk.T_base_tcp[:, :3, :3])).as_rotvec()
    turns = savgol_filter(turns, window, _SMOOTH_ORDER, axis=0, mode="interp")
    poses[:, :3, :3] = (first * Rotation.from_rotvec(turns)).as_matrix()
    return PoseChunk(t=chunk.t, T_base_tcp=poses, grip=chunk.grip)


def _sample(plan: PoseChunk, t: float) -> tuple[npt.NDArray[np.float64], float]:
    """`plan` at time `t`, which lies inside it."""
    if len(plan.t) == 1:
        return plan.T_base_tcp[0].copy(), float(plan.grip[0])
    i = min(int(np.searchsorted(plan.t, t, side="right")) - 1, len(plan.t) - 2)
    alpha = (t - plan.t[i]) / (plan.t[i + 1] - plan.t[i])
    pose = _between(plan.T_base_tcp[i], plan.T_base_tcp[i + 1], alpha)
    return pose, float((1 - alpha) * plan.grip[i] + alpha * plan.grip[i + 1])


def _between(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], share: float) -> npt.NDArray[np.float64]:
    """The pose `share` of the way from `a` to `b`: the position on the straight line, the rotation on the shortest
    arc."""
    pose = np.eye(4)
    rotation_a = Rotation.from_matrix(a[:3, :3])
    turn = (rotation_a.inv() * Rotation.from_matrix(b[:3, :3])).as_rotvec()
    pose[:3, :3] = (rotation_a * Rotation.from_rotvec(share * turn)).as_matrix()
    pose[:3, 3] = (1 - share) * a[:3, 3] + share * b[:3, 3]
    return pose


def _old_weight(x: float) -> float:
    """The old plan's weight `x` of the way through the blend window, rescaled to be 1 at x = 0 and 0 at x = 1."""

    def sigmoid(u: float) -> float:
        return 1.0 / (1.0 + np.exp(_SIGMOID_STEEPNESS * (u - 0.5)))

    return float((sigmoid(x) - sigmoid(1.0)) / (sigmoid(0.0) - sigmoid(1.0)))
