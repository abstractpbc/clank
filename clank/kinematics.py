"""Forward kinematics and differential IK for one arm, built from its URDF."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pink
import pinocchio as pin
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask, PostureTask

_TCP_FRAME = "tcp"
_QP_SOLVER = "daqp"


class Kinematics:
    def __init__(
        self,
        urdf: Path,
        joints: Sequence[str],
        tcp_link: str,
        tcp_offset: Sequence[float],
        tcp_rotation: Sequence[Sequence[float]],
        rest: Sequence[float],
        limit_tolerance_rad: float,
    ) -> None:
        """
        urdf: the arm's URDF; its root link is the arm's base frame.
        joints: the joints the arm drives, in the order the kinematic chain has them; every other joint is locked
            at its neutral position.
        tcp_link: the link the tool centre point is fixed to.
        tcp_offset: (x, y, z) of the tool centre point in `tcp_link`'s frame, metres.
        tcp_rotation: the tool centre point's axes in `tcp_link`'s frame, a 3 x 3 rotation matrix; the identity keeps
            that link's axes.
        rest: one angle per joint, the posture the solve leans toward where the tool pose leaves it free.
        limit_tolerance_rad: how far past a URDF joint limit a measured joint may read (an arm resting on a hard
            stop reads slightly past it); beyond this, ik_step() refuses.
        """
        full = pin.buildModelFromUrdf(str(urdf))
        missing = [name for name in joints if not full.existJointName(name)]
        if missing:
            raise ValueError(f"{urdf} has no joints named {missing}; it has {list(full.names[1:])}")
        locked = [full.getJointId(name) for name in full.names[1:] if name not in joints]
        model = pin.buildReducedModel(full, locked, pin.neutral(full))
        if list(model.names[1:]) != list(joints):
            raise ValueError(f"joints must follow the kinematic chain's order, {list(model.names[1:])}")
        if not model.nq == model.nv == len(joints):
            raise ValueError("every driven joint must have one degree of freedom with limits")
        if not model.existFrame(tcp_link):
            raise ValueError(f"{urdf} has no link named {tcp_link!r}")
        link_id = model.getFrameId(tcp_link)
        link = model.frames[link_id]
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(rotation), 1.0):
            raise ValueError(f"tcp_rotation must be a 3 x 3 rotation matrix, not {rotation.tolist()}")
        placement = link.placement * pin.SE3(rotation, np.asarray(tcp_offset, dtype=np.float64))
        self._tcp_id = model.addFrame(pin.Frame(_TCP_FRAME, link.parentJoint, link_id, placement, pin.FrameType.OP_FRAME))
        self._model = model
        # Joint position limits only: nothing caps joint speed, so a step goes as far as the target pose asks.
        self._limits = (ConfigurationLimit(model),)
        self._data = model.createData()
        self._joints = tuple(joints)
        self._tolerance = limit_tolerance_rad
        self._tcp_task = FrameTask(_TCP_FRAME, position_cost=1.0, orientation_cost=1.0)
        posture = PostureTask(cost=1e-3)
        posture.set_target(np.asarray(rest, dtype=np.float64))
        self._tasks = (self._tcp_task, posture)

    def fk(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The tool pose (4, 4) in the arm's base frame at joints `q`. Safe to call from any thread: it works in its
        own scratch space, since the run's observer and its control loop both ask."""
        data = self._model.createData()
        pin.framesForwardKinematics(self._model, data, np.asarray(q, dtype=np.float64))
        return data.oMf[self._tcp_id].homogeneous.copy()

    def ik_step(self, q: npt.NDArray[np.float64], T_base_tcp: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """The joints to command next: one step from `q` toward the tool pose, inside the joint limits. There is no
        speed cap; `dt` only scales the solve.

        Raises ValueError when `q` is past a joint limit by more than the tolerance, and pink's NoSolutionFound when
        the step has no solution.
        """
        q = np.asarray(q, dtype=np.float64)
        low, high = self._model.lowerPositionLimit, self._model.upperPositionLimit
        outside = (q < low - self._tolerance) | (q > high + self._tolerance)
        if np.any(outside):
            where = {name: float(q[i]) for i, name in enumerate(self._joints) if outside[i]}
            raise ValueError(f"joints past their limits by more than {self._tolerance} rad: {where}")
        configuration = pink.Configuration(self._model, self._data, q)
        self._tcp_task.set_target(pin.SE3(T_base_tcp[:3, :3], T_base_tcp[:3, 3]))
        # safety_break is off because a joint inside the tolerance band is allowed; the limits still bound the step.
        velocity = pink.solve_ik(
            configuration, self._tasks, dt, solver=_QP_SOLVER, limits=self._limits, safety_break=False
        )
        return configuration.integrate(velocity, dt)
