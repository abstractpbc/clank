"""Camera video kept only while the robot moves: one H.264 MP4 per camera, encoded on its own thread."""

import json
import queue
import threading
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from clank.types import JointState

## How far a joint must move, radians, or a gripper's aperture change, from where the arm last moved, to count as motion.
MOVE_RAD = 0.01
MOVE_GRIP = 0.02
## How long the video runs on after the last motion, seconds.
HOLD_S = 1.0
## How many frames may wait for the encoder, across all cameras, before new ones are dropped.
QUEUE_FRAMES = 120
## libx264's speed preset and quality: `veryfast` keeps three cameras well inside one core's share on one station.
PRESET = "veryfast"
CRF = 23
## The gap left in the video where idle time was cut out, milliseconds: one frame at the cameras' usual 15 fps.
CUT_MS = 67
## The videos' timestamps are in milliseconds; the rate a player is told is nominal, the timestamps are what count.
_TIME_BASE = Fraction(1, 1000)
_NOMINAL_FPS = 15


class MotionRecorder:
    def __init__(self, out: Path, roles: tuple[str, ...], started: float) -> None:
        """out is a new directory for the videos, roles the camera roles, started the run's monotonic start."""
        out.mkdir(parents=True)
        self._out = out
        self._roles = roles
        self._started = started
        self._anchors: dict[str, JointState] = {}
        ## When an arm last moved on the monotonic clock; None until one has. Written by the control loop's thread
        ## only and read by the camera loop's; a float attribute is read and written whole.
        self._last_motion: float | None = None
        self._segments: list[list[float]] = []
        self._queue: queue.Queue[tuple[str, npt.NDArray[np.uint8], float] | None] = queue.Queue(QUEUE_FRAMES)
        self._dropped = {role: 0 for role in roles}
        self._kept = {role: 0 for role in roles}
        self._error: BaseException | None = None
        self._encoder = threading.Thread(target=self._encode, name="clank-recorder")
        self._encoder.start()

    def note_arms(self, now: float, states: Mapping[str, JointState]) -> None:
        """Takes the arms' measured state; call it on every control tick."""
        for side, state in states.items():
            anchor = self._anchors.get(side)
            if anchor is None:
                self._anchors[side] = state
                continue
            if np.abs(state.q - anchor.q).max() > MOVE_RAD or abs(state.grip - anchor.grip) > MOVE_GRIP:
                self._anchors[side] = state
                self._last_motion = now

    def offer(self, frames: Mapping[str, npt.NDArray[np.uint8]], t: float) -> None:
        """Takes one camera tick, captured at `t` on the monotonic clock; keeps it if the robot is moving."""
        if self._error is not None:
            raise RuntimeError("the recorder's encoder failed") from self._error
        last = self._last_motion
        if last is None or t - last > HOLD_S:
            return
        run_t = t - self._started
        if not self._segments or run_t - self._segments[-1][1] > HOLD_S:
            self._segments.append([run_t, run_t])
        self._segments[-1][1] = run_t
        for role, frame in frames.items():
            try:
                self._queue.put_nowait((role, frame, run_t))
            except queue.Full:
                self._dropped[role] += 1

    def close(self) -> dict:
        """Finishes the videos and returns frames kept and dropped per camera and the stretches of motion."""
        self._queue.put(None)
        self._encoder.join()
        summary = {
            "kept_frames": self._kept,
            "dropped_frames": self._dropped,
            "segments_s": [[round(start, 3), round(end, 3)] for start, end in self._segments],
            "seconds_recorded": round(sum(end - start for start, end in self._segments), 2),
        }
        (self._out / "segments.json").write_text(json.dumps(summary, indent=2))
        if self._error is not None:
            raise RuntimeError("the recorder's encoder failed") from self._error
        return summary

    def _encode(self) -> None:
        """The encoder thread: writes every queued frame into its camera's video until it is handed None."""
        containers: dict[str, av.container.OutputContainer] = {}
        streams: dict[str, av.video.stream.VideoStream] = {}
        times: dict[str, list[float]] = {role: [] for role in self._roles}
        last_pts: dict[str, int] = {}
        try:
            while (item := self._queue.get()) is not None:
                role, frame, run_t = item
                if role not in containers:
                    containers[role] = av.open(str(self._out / f"{role}.mp4"), mode="w")
                    stream = containers[role].add_stream("libx264", rate=_NOMINAL_FPS)
                    stream.width, stream.height = frame.shape[1], frame.shape[0]
                    stream.pix_fmt = "yuv420p"
                    stream.time_base = _TIME_BASE
                    # The encoder's own clock must be milliseconds too: left unset it takes 1 / the nominal rate,
                    # and two frames closer than that share a timestamp, which the MP4 writer refuses.
                    stream.codec_context.time_base = _TIME_BASE
                    stream.options = {"preset": PRESET, "crf": str(CRF)}
                    streams[role] = stream
                previous = times[role][-1] if times[role] else None
                if previous is None:
                    pts = 0
                elif run_t - previous > HOLD_S:
                    # A new stretch of motion: it follows the last one a single frame later, so no idle time plays.
                    pts = last_pts[role] + CUT_MS
                else:
                    pts = last_pts[role] + max(round(1000.0 * (run_t - previous)), 1)
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts = pts
                video_frame.time_base = _TIME_BASE
                for packet in streams[role].encode(video_frame):
                    containers[role].mux(packet)
                last_pts[role] = pts
                times[role].append(run_t)
                self._kept[role] += 1
            for role, stream in streams.items():
                for packet in stream.encode():
                    containers[role].mux(packet)
        except BaseException as error:
            self._error = error
            # Keep taking frames until close() hands None, so the camera loop is never held up by a full queue.
            while self._queue.get() is not None:
                pass
        finally:
            for container in containers.values():
                container.close()
            for role, kept in times.items():
                np.save(self._out / f"{role}_t.npy", np.array(kept))
