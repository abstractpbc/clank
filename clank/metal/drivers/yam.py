"""One i2rt YAM arm with a linear 4310 gripper, powered and commanded through i2rt.

Opening the arm switches its motors on, and i2rt sweeps the gripper to both hard stops to calibrate it; the arm then
holds where it is. command() hands i2rt a position target, which its own loop tracks with its joint gains; i2rt clips
the target to its joint limits. Nothing here limits how fast a target may change.

A YAM has no brakes: close() switches the motors off and the arm goes limp, so park it first.
"""

import time

import numpy as np
import numpy.typing as npt
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

from clank.types import JointState

_ARM_JOINTS = 6


class YamRobot:
    def __init__(self, channel: str) -> None:
        """channel: the arm's SocketCAN interface, e.g. `can0`."""
        self._channel = channel
        self._robot = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)

    def read(self) -> JointState:
        self._require_alive()
        position = self._robot.get_joint_pos()  # six joints, then the gripper from 0 closed to 1 open
        return JointState(t=time.monotonic(), q=position[:_ARM_JOINTS].copy(), grip=float(position[_ARM_JOINTS]))

    def command(self, q: npt.NDArray[np.float64], grip: float) -> None:
        self._require_alive()
        target = np.append(np.asarray(q, dtype=np.float64), grip)
        if target.shape != (_ARM_JOINTS + 1,) or not np.all(np.isfinite(target)):
            raise ValueError(f"a YAM takes six finite joint angles and a gripper aperture, got {target}")
        self._robot.command_joint_pos(target)

    def close(self) -> None:
        self._robot.close()

    def _require_alive(self) -> None:
        """Refuses once i2rt's motor loop has stopped (a motor stopped answering, as when power is cut): from then
        on i2rt returns the last state it had and drops every target, so the arm is neither read nor driven."""
        if not self._robot.motor_chain.running:
            raise RuntimeError(f"{self._channel}: i2rt's motor loop has stopped; the arm is no longer driven")
