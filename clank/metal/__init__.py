"""Hardware: everything that touches an arm, a motor or a camera. `load_workcell("<workcell id>")` returns a robot
station.

A driver (`clank/metal/drivers/`) is the code that talks to one kind of device, behind the Robot or Camera
interface. A workcell (`clank/metal/workcells/<workcell id>.py`) is a declaration: one station's arms and cameras, each
naming the driver that opens it, and each arm's safety envelope (`clank/metal/safety/`): the speed limits and workspace
its commands are held to.
"""

from clank.metal import workcells
from clank.metal.camera import Camera
from clank.metal.robot import Robot
from clank.metal.safety import Box, Envelope
from clank.metal.workcell import Arm, Workcell
from clank.registry import load_declared

__all__ = ["Arm", "Box", "Camera", "Envelope", "Robot", "Workcell", "load_workcell"]


def load_workcell(workcell_id: str) -> Workcell:
    """Imports `clank.metal.workcells.<workcell_id>` and returns the workcell it declares.

    A workcell is imported only when it is asked for, and it opens nothing until one of its `open` functions is called,
    so naming a workcell needs neither its hardware nor its drivers' libraries.
    """
    return load_declared(workcells, workcell_id, "workcell")
