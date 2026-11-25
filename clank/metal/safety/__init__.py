"""The safety envelope: a workspace box and joint speed limits, which are no substitute for an e-stop."""

from clank.metal.safety.envelope import Envelope
from clank.metal.safety.workspace import Box

__all__ = ["Box", "Envelope"]
