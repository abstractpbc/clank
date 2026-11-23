"""The safety envelope: the limits an arm's commands are held to, whatever its model asks.

Every arm of a workcell declares one (`Arm.safety`), and the run applies it on every tick:

- `workspace.py`: a box in the arm's base frame. A tool target outside it is moved to its nearest point inside before
  the IK solves for it. No arm is powered while its tool is outside its box, and the run stops when a commanded tool
  strays past it by more than a centimetre.
- `speed.py`: each joint's speed limit. Each step the IK takes is shortened, as a whole, so that no joint goes faster
  than its limit, and the return to rest takes as long as the limits need.

What the run did to stay inside is counted in its `stats.json`: the ticks a target was moved into the box and the
ticks a step was shortened. A limit that is reached often says the model wants something the envelope forbids.

The envelope holds what clank sends. It is no substitute for an e-stop, and it cannot stop an arm that is pushed,
falls, or is driven by something else.
"""

from clank.metal.safety.envelope import Envelope
from clank.metal.safety.workspace import Box

__all__ = ["Box", "Envelope"]
