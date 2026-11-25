"""The moves a language model drives arms with, in metres in each arm's base frame, apertures 0 to 1."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from clank.ai.protocols.nerve.llm import Tool


@dataclass(frozen=True)
class ArmTarget:
    """Where one arm is told to be: its tool pose (4, 4) in its base frame, and its gripper aperture."""

    T_base_tcp: npt.NDArray[np.float64]
    grip: float


@dataclass(frozen=True)
class Move:
    """One arm's next target, and how long it takes to get there, seconds."""

    target: ArmTarget
    seconds: float


@dataclass(frozen=True)
class Scene:
    """What a primitive reads: where each arm was last told to be, and where it began the episode."""

    targets: Mapping[str, ArmTarget]
    home: Mapping[str, ArmTarget]


@dataclass(frozen=True)
class Outcome:
    """What a primitive decided: what to tell the model, each arm's move, and whether the task is done."""

    text: str
    moves: Mapping[str, Move] = field(default_factory=dict)
    done: bool = False


@dataclass(frozen=True)
class Primitive:
    tool: Tool
    run: Callable[[Scene, Mapping[str, Any]], Outcome]


class Refused(Exception):
    """The model's call cannot be carried out as asked. The message goes back to the model as the call's result."""


def _move_to(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    side, now = _arm(scene, arguments)
    pose = now.T_base_tcp.copy()
    pose[:3, 3] = [_number(arguments, name) for name in ("x", "y", "z")]
    return _moved(side, ArmTarget(pose, now.grip), arguments)


def _move_by(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    side, now = _arm(scene, arguments)
    pose = now.T_base_tcp.copy()
    pose[:3, 3] += [_number(arguments, name) for name in ("dx", "dy", "dz")]
    return _moved(side, ArmTarget(pose, now.grip), arguments)


def _turn(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    side, now = _arm(scene, arguments)
    axis = arguments.get("axis")
    if axis not in ("x", "y", "z"):
        raise Refused(f"axis must be x, y or z, got {axis!r}")
    pose = now.T_base_tcp.copy()
    turn = Rotation.from_euler(axis, _number(arguments, "degrees"), degrees=True).as_matrix()
    pose[:3, :3] = turn @ pose[:3, :3]  # about the base frame's axis, with the tool point staying where it is
    return _moved(side, ArmTarget(pose, now.grip), arguments)


def _grip(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    side, now = _arm(scene, arguments)
    aperture = _number(arguments, "aperture")
    if not 0.0 <= aperture <= 1.0:
        raise Refused(f"aperture must be between 0 (closed) and 1 (open), got {aperture}")
    return _moved(side, ArmTarget(now.T_base_tcp, aperture), arguments)


def _home(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    side, _now = _arm(scene, arguments)
    return _moved(side, scene.home[side], arguments)


def _done(scene: Scene, arguments: Mapping[str, Any]) -> Outcome:
    summary = arguments.get("summary")
    if not isinstance(summary, str) or not summary:
        raise Refused("summary must say, in words, what was done or why the task cannot be")
    return Outcome(text="The task is closed. The arms hold where they are.", done=True)


def _arm(scene: Scene, arguments: Mapping[str, Any]) -> tuple[str, ArmTarget]:
    side = arguments.get("arm")
    if side not in scene.targets:
        raise Refused(f"arm must be one of {sorted(scene.targets)}, got {side!r}")
    return side, scene.targets[side]


def _number(arguments: Mapping[str, Any], name: str) -> float:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise Refused(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _moved(side: str, target: ArmTarget, arguments: Mapping[str, Any]) -> Outcome:
    seconds = _number(arguments, "seconds")
    if seconds <= 0:
        raise Refused(f"seconds must be positive, got {seconds}")
    x, y, z = target.T_base_tcp[:3, 3]
    return Outcome(
        text=f"The {side} arm is moving to x={x:.3f} y={y:.3f} z={z:.3f} m, gripper {target.grip:.2f}, over {seconds:g} s.",
        moves={side: Move(target, seconds)},
    )


def _tool(name: str, description: str, numbers: Mapping[str, str], **others: Mapping[str, Any]) -> Tool:
    """A tool whose arguments are numbers (name -> what it means) plus any others, all required."""
    properties = {field_name: {"type": "number", "description": meaning} for field_name, meaning in numbers.items()}
    properties.update(others)
    return Tool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
    )


_ARM = {"type": "string", "description": "which arm"}
_SECONDS = {"seconds": "how long the move takes; a move faster than the motors can follow stops the run"}

PRIMITIVES: tuple[Primitive, ...] = (
    Primitive(
        _tool(
            "move_to",
            "Move an arm's tool to a position in that arm's base frame, in a straight line, keeping its orientation.",
            {"x": "metres", "y": "metres", "z": "metres", **_SECONDS},
            arm=_ARM,
        ),
        _move_to,
    ),
    Primitive(
        _tool(
            "move_by",
            "Move an arm's tool by an offset along its base frame's axes, in a straight line, keeping its orientation.",
            {"dx": "metres", "dy": "metres", "dz": "metres", **_SECONDS},
            arm=_ARM,
        ),
        _move_by,
    ),
    Primitive(
        _tool(
            "turn",
            "Turn an arm's tool about one axis of its base frame, with the tool point staying where it is.",
            {"degrees": "how far to turn, right-handed about the axis", **_SECONDS},
            arm=_ARM,
            axis={"type": "string", "enum": ["x", "y", "z"], "description": "the base frame axis to turn about"},
        ),
        _turn,
    ),
    Primitive(
        _tool(
            "grip",
            "Open or close an arm's gripper, with the arm staying where it is.",
            {"aperture": "0 is closed, 1 is open", **_SECONDS},
            arm=_ARM,
        ),
        _grip,
    ),
    Primitive(
        _tool(
            "home",
            "Return an arm to where it was when the task began, tool pose and gripper.",
            dict(_SECONDS),
            arm=_ARM,
        ),
        _home,
    ),
    Primitive(
        _tool(
            "done",
            "Close the task: it is complete, or it cannot be done.",
            {},
            summary={"type": "string", "description": "what was done, or why the task cannot be"},
        ),
        _done,
    ),
)
