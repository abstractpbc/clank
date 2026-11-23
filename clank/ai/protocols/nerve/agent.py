"""The agent loop: a language model in clank's Policy slot.

One turn per infer(). The model is shown the cameras and where each arm's tool is, calls primitives
(`primitives.py`), and each arm's move becomes a pose chunk: a straight line from where the arm is to the target,
then a hold. While the arms are moving, infer() says the model is not ready, so the next turn starts from arms that
have stopped and shows the model what its last calls did.

Arms named in the same turn move together. Each arm takes one move a turn.
"""

import dataclasses
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from clank.ai.policy import PolicyNotReady
from clank.ai.protocols.nerve.llm import LLM, Call, Turn
from clank.ai.protocols.nerve.primitives import PRIMITIVES, ArmTarget, Move, Primitive, Refused, Scene
from clank.types import JointState, PoseChunk

SYSTEM = (
    "You control robot arms by calling the primitives you are given. Each turn you are shown the cameras and where "
    "each arm's tool is. Call primitives to act; you see what they did on the next turn. Positions are metres in "
    "each arm's own base frame, and gripper apertures run from 0 (closed) to 1 (open). Arms named in the same turn "
    "move together, and each arm takes one move a turn. Call done when the task is complete or cannot be done."
)
## Camera frames are sent no wider than this, as JPEGs of this quality.
IMAGE_WIDTH = 640
JPEG_QUALITY = 80
## After a move's time is up, this long passes before the model is asked again, so the frames it sees are of arms
## that have stopped.
SETTLE_S = 0.25


class NervePolicy:
    def __init__(self, config: Mapping[str, Any], llm: LLM, primitives: Sequence[Primitive] = PRIMITIVES) -> None:
        """
        config: `{"think_s": ...}`: how long the arms hold their last target waiting for the model's next turn,
            seconds. It has to cover the model's slowest answer: clank stops a run whose plan runs out.
        llm: the language model.
        primitives: what the model may call.
        """
        self._think_s = float(config["think_s"])
        if not self._think_s > 0:
            raise ValueError(f"think_s must be positive, got {self._think_s}")
        self._llm = llm
        self._primitives = {primitive.tool.name: primitive for primitive in primitives}
        self._lock = threading.Lock()
        self._closed = False
        self.reset_episode()

    def observe(
        self,
        frames: Mapping[str, npt.NDArray[np.uint8]],
        states: Mapping[str, JointState],
        poses: Mapping[str, npt.NDArray[np.float64]],
        prompt: str,
        t: float,
    ) -> None:
        with self._lock:
            self._refuse_closed()
            self._latest = (frames, states, poses, prompt, t)

    def infer(self) -> dict[str, PoseChunk]:
        with self._lock:
            self._refuse_closed()
            latest = self._latest
        if latest is None:
            raise RuntimeError("infer() was called before any observe()")
        waiting = self._busy_until - time.monotonic()
        if waiting > 0:
            raise PolicyNotReady("the arms are still moving", waiting)
        frames, states, poses, prompt, t_obs = latest
        if self._targets is None:
            # The episode starts here: every arm is told to be where it is.
            self._targets = {side: ArmTarget(poses[side], states[side].grip) for side in sorted(poses)}
            self._home = dict(self._targets)
            self._turns = [Turn("system", SYSTEM)]
        before = dict(self._targets)
        moves: dict[str, Move] = {}
        if not self._done:
            self._turns.append(_observation(frames, states, poses, prompt))
            answer = self._llm.reply(_latest_images_only(self._turns), [p.tool for p in self._primitives.values()])
            self._turns.append(answer)
            for call in answer.calls:
                self._turns.append(Turn("tool", self._run(call, moves), call_id=call.id))
        for side, move in moves.items():
            self._targets[side] = move.target
        # The moves start now, not at the observation: the arms stood still while the model thought.
        start = time.monotonic()
        longest = max((move.seconds for move in moves.values()), default=0.0)
        self._busy_until = start + longest + (SETTLE_S if moves else 0.0)
        hold_until = start + longest + SETTLE_S + self._think_s
        chunks = {}
        for side, target in self._targets.items():
            arrive = start + moves[side].seconds if side in moves else start
            # Four rows. Where the arm was at the observation, which clank drops as already past. The target when
            # the move's time is up. The target again a moment later: clank fades a new chunk in over a short
            # window, and this row, past that window, pins the arm to the target even after a very short move.
            # The target once more at the end of the hold.
            rows = (before[side], target, target, target)
            chunks[side] = PoseChunk(
                t=np.array([t_obs, arrive, arrive + SETTLE_S, hold_until]),
                T_base_tcp=np.stack([row.T_base_tcp for row in rows]),
                grip=np.array([row.grip for row in rows]),
            )
        return chunks

    def _run(self, call: Call, moves: dict[str, Move]) -> str:
        """Carries out one of the model's calls, adding its moves to `moves`; returns what to tell the model."""
        primitive = self._primitives.get(call.name)
        if primitive is None:
            return f"Refused: there is no primitive {call.name!r}; the primitives are {sorted(self._primitives)}."
        try:
            outcome = primitive.run(Scene(targets=self._targets, home=self._home), call.arguments)
        except Refused as refusal:
            return f"Refused: {refusal}."
        twice = sorted(set(outcome.moves) & set(moves))
        if twice:
            return f"Refused: the {twice[0]} arm already has a move this turn; call again next turn."
        moves.update(outcome.moves)
        self._done = self._done or outcome.done
        return outcome.text

    def reset_episode(self) -> None:
        with self._lock:
            self._refuse_closed()
            self._latest = None
        self._targets: dict[str, ArmTarget] | None = None
        self._home: dict[str, ArmTarget] = {}
        self._turns: list[Turn] = []
        self._done = False
        self._busy_until = 0.0

    def _refuse_closed(self) -> None:
        if self._closed:
            raise RuntimeError("the Nerve policy is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._latest = None
        self._llm.close()


def _observation(
    frames: Mapping[str, npt.NDArray[np.uint8]],
    states: Mapping[str, JointState],
    poses: Mapping[str, npt.NDArray[np.float64]],
    prompt: str,
) -> Turn:
    """What the model is shown: the task, each arm's measured tool position and gripper, and every camera."""
    lines = [f"Task: {prompt}"]
    for side in sorted(poses):
        x, y, z = poses[side][:3, 3]
        lines.append(f"{side} arm: tool at x={x:.3f} y={y:.3f} z={z:.3f} m, gripper {states[side].grip:.2f}")
    roles = sorted(frames)
    lines.append(f"The images are, in order: {', '.join(roles)}.")
    return Turn("user", "\n".join(lines), images=tuple(_jpeg(frames[role]) for role in roles))


def _jpeg(image: npt.NDArray[np.uint8]) -> bytes:
    """One (H, W, 3) BGR frame as a JPEG no wider than `IMAGE_WIDTH`."""
    height, width = image.shape[:2]
    if width > IMAGE_WIDTH:
        image = cv2.resize(image, (IMAGE_WIDTH, round(height * IMAGE_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("OpenCV could not encode the frame as JPEG")
    return jpeg.tobytes()


def _latest_images_only(turns: Sequence[Turn]) -> list[Turn]:
    """The conversation with images kept on the newest observation only: older ones keep their words."""
    newest = max(i for i, turn in enumerate(turns) if turn.role == "user")
    return [dataclasses.replace(turn, images=()) if turn.images and i != newest else turn for i, turn in enumerate(turns)]
