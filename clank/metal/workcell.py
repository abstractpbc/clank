"""What a workcell file declares: one robot station, with its arms and its cameras."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from clank.kinematics import Kinematics
from clank.metal.camera import Camera
from clank.metal.robot import Robot
from clank.metal.safety import Envelope


@dataclass(frozen=True)
class Arm:
    """One arm: how to open it powered or passive, its kinematics, rest joints, top speeds, and safety envelope."""

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
    """A robot station: id (its file's name), display name, arms by side, cameras by role, rest aperture."""

    id: str
    name: str
    arms: Mapping[str, Arm]
    cameras: Mapping[str, Callable[[], Camera]]
    rest_grip: float
