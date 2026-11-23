"""The recording protocol: a model that is a recorded run, played back, with no service behind it.

A live run saves every chunk its model answered, and when each arrived, in the run directory's `log.npz`. This plays
those chunks back in order. Each infer() waits as long as the recorded answer took to come, then returns the recorded
chunk, restamped to start at the observation it is now answering. Nothing is read from an observation but its time,
so the plan is the recorded one whatever the arms are doing: an arm that starts where the recorded run started
repeats that run's motion.
"""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from clank.types import JointState, PoseChunk

_ROWS_SUFFIX = "_chunk_t"


class RecordingPolicy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        """config: `{"recording": <path of a run's log.npz>}`."""
        self._path = Path(config["recording"])
        with np.load(self._path) as log:
            sides = sorted(name.removesuffix(_ROWS_SUFFIX) for name in log.files if name.endswith(_ROWS_SUFFIX))
            if "chunk_arrived" not in log.files or not sides or len(log["chunk_arrived"]) == 0:
                raise ValueError(f"{self._path} holds no recorded chunks; only a live run that logged them can be played")
            ## When each chunk arrived, on the recorded run's clock: (chunks,).
            self._arrived = log["chunk_arrived"]
            ## Per side, each chunk's row times (chunks, rows), tool poses (chunks, rows, 4, 4) and apertures.
            self._chunks = {
                side: (log[f"{side}_chunk_t"], log[f"{side}_chunk_poses"], log[f"{side}_chunk_grip"]) for side in sides
            }
        self._next = 0
        self._t_obs: float | None = None
        self._closed = False
        ## How long the last infer() took to answer: the recorded chunk's wait, in the shape a live model reports.
        self.last_timing: dict[str, float] | None = None

    def observe(
        self,
        frames: Mapping[str, npt.NDArray[np.uint8]],
        states: Mapping[str, JointState],
        poses: Mapping[str, npt.NDArray[np.float64]],
        prompt: str,
        t: float,
    ) -> None:
        self._refuse_closed()
        self._t_obs = t

    def infer(self) -> dict[str, PoseChunk]:
        self._refuse_closed()
        t_obs = self._t_obs
        if t_obs is None:
            raise RuntimeError("infer() was called before any observe()")
        index = self._next
        if index == len(self._arrived):
            raise RuntimeError(f"all {index} chunks of {self._path} have been played")
        observed = {side: rows[index][0] for side, (rows, _poses, _grip) in self._chunks.items()}
        waited = float(self._arrived[index] - min(observed.values()))
        time.sleep(waited)
        self._next = index + 1
        self.last_timing = {"rtt_ms": 1000.0 * waited}
        return {
            side: PoseChunk(
                t=t_obs + (rows[index] - observed[side]),
                T_base_tcp=poses[index].copy(),
                grip=grip[index].copy(),
            )
            for side, (rows, poses, grip) in self._chunks.items()
        }

    def reset_episode(self) -> None:
        """Starts the recording over from its first chunk."""
        self._refuse_closed()
        self._next = 0
        self._t_obs = None

    def _refuse_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"the recording {self._path} is closed")

    def close(self) -> None:
        self._closed = True
        self._t_obs = None
