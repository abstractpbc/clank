"""What a workcell file declares: one robot station, with its arms and its cameras."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from clank.kinematics import Kinematics
from clank.metal.camera import Camera
from clank.metal.robot import Robot
from clank.metal.safety import Envelope


@dataclass(frozen=True)
class Arm:
    """One arm of a workcell.

    open: powers the arm and returns its driver.
    open_passive: opens the arm to be read without powering it, for the moments before a run commits to moving.
        Called with the gripper aperture to report, which such a read cannot measure.
    kinematics: builds the arm's kinematic model: the one the tool poses it is sent are expressed in.
    rest_q: the joints the arm rests at, where a run leaves it.
    top_speed: each joint's top speed, rad/s: what its motor can do. No speed limit in `safety` may exceed it.
    safety: the limits the arm's commands are held to (`clank/metal/safety/`): a speed limit per joint, no higher than
        its `top_speed`, and the tool's workspace.
    """

    open: Callable[[], Robot]
    open_passive: Callable[[float], Robot]
    kinematics: Callable[[], Kinematics]
    rest_q: tuple[float, ...]
    top_speed: tuple[float, ...]
    safety: Envelope

    def __post_init__(self) -> None:
        if len(self.safety.speed_limit) != len(self.top_speed):
            raise ValueError(
                f"the arm has {len(self.top_speed)} joints' top speeds and {len(self.safety.speed_limit)} speed limits"
            )
        over = [i for i, (limit, top) in enumerate(zip(self.safety.speed_limit, self.top_speed)) if limit > top]
        if over:
            raise ValueError(f"joints {over} have a speed limit above what their motors can do")


@dataclass(frozen=True)
class Workcell:
    """A robot station.

    id: the workcell's name, which is also its file's: `clank/metal/workcells/<id>.py`.
    name: what to call the workcell in a message to a person.
    arms: side -> arm.
    cameras: camera role -> a function that opens that camera.
    rest_grip: the gripper aperture at rest, 0 closed to 1 open.
    """

    id: str
    name: str
    arms: Mapping[str, Arm]
    cameras: Mapping[str, Callable[[], Camera]]
    rest_grip: float
