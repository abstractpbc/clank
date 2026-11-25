"""Hardware: drivers, workcells and safety envelopes; `load_workcell("<workcell id>")` returns a station."""

from clank.metal import workcells
from clank.metal.camera import Camera
from clank.metal.robot import Robot
from clank.metal.safety import Box, Envelope
from clank.metal.workcell import Arm, Workcell
from clank.registry import load_declared

__all__ = ["Arm", "Box", "Camera", "Envelope", "Robot", "Workcell", "load_workcell"]


def load_workcell(workcell_id: str) -> Workcell:
    """The workcell `clank.metal.workcells.<workcell_id>` declares, imported only when asked for."""
    return load_declared(workcells, workcell_id, "workcell")
