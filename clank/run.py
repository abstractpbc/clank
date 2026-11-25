"""A model driving a workcell: `python -m clank.run --policy <provider>/<model> --workcell <workcell> ...`."""

import argparse
import contextlib
import json
import signal
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from clank.ai import Policy, PolicyNotReady, create_policy, find_model
from clank.kinematics import Kinematics
from clank.metal import Camera, Robot, Workcell, load_workcell
from clank.recorder import MotionRecorder
from clank.trajectory import Trajectory
from clank.types import JointState, PoseChunk

CONTROL_HZ = 100.0
## How often the model is asked for a chunk, from one ask to the next. Asking back to back gets a hosted model
## rate-limited, and every chunk is one more seam.
REPLAN_S = 1.0
## How long a new chunk takes to fade in from the plan in force (five rows at 20 Hz),
## for a model that does not set its own (`Model.blend_s`).
BLEND_S = 0.25
## The joints sent to the motors follow the IK's through a critically damped low-pass at this cutoff, which takes out
## chatter the plan or the solve leaves, for about 2 / (2 pi f) = 64 ms of lag.
COMMAND_FILTER_HZ = 5.0
## How long the model is fed observations before it is first asked, so one that reads a window of frames has it.
WINDOW_FILL_S = 2.0
## The return to rest: a straight line in joint space over this long, in this many steps a second.
PARK_S = 4.0
PARK_HZ = 50.0
## The run stops when an arm joint stays this far from where it was told to be for this long: the arm is blocked or
## no longer following. A joint moving fast trails its target by more than this for a moment, which is not that.
FOLLOW_LIMIT_RAD = 0.5
FOLLOW_PATIENCE_S = 0.5
## The run stops when the IK wants a joint further than this from where its speed limits let it be. A plan faster than
## the limits only builds a lead while it outruns them, and the arm catches up when it slows; a lead this large is a
## plan seconds ahead of the arm, or an IK that flipped to another solution, which the arm would otherwise swing to.
PLAN_LEAD_LIMIT_RAD = 1.0
## How far outside its workspace an arm's commanded tool may be before the run stops. The command filter bends the path
## the IK planned by millimetres, so a tool sliding along a face of the box dips just past it.
WORKSPACE_SLACK_M = 0.01
## How long the arms may hold the end of a plan that ran out, waiting for the model's next chunk, before the run stops.
## Before any plan is in force, how long chunks may keep arriving too late to use.
PLAN_GAP_S = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", required=True, help="the model, as <provider id>/<model id>")
    parser.add_argument("--workcell", required=True, help="the robot station, by its workcell id")
    parser.add_argument("--prompt", required=True, help="the task, in words")
    parser.add_argument("--seconds", required=True, type=float, help="how long the model drives the arms")
    parser.add_argument("--max-wait-s", required=True, type=float, help="how long to wait for the model to start")
    parser.add_argument("--replan-s", type=float, default=REPLAN_S, help="seconds between requests for a new chunk (default: 1)")
    parser.add_argument("--out", required=True, type=Path, help="a new directory for the run's log")
    args = parser.parse_args()
    if not np.isfinite(args.replan_s) or args.replan_s <= 0:
        parser.error("--replan-s must be finite and positive")
    # Both names are looked up before anything is created or opened, so a mistyped one leaves nothing behind.
    workcell = load_workcell(args.workcell)
    _, model = find_model(args.policy)
    blend_s = BLEND_S if model.blend_s is None else model.blend_s
    args.out.mkdir(parents=True)
    # Ctrl-C and a plain `kill` both end the run through the return to rest, even when started in the background.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    kinematics = {side: arm.kinematics() for side, arm in workcell.arms.items()}
    for side, arm in workcell.arms.items():
        arm.safety.require_inside(kinematics[side].fk(np.array(arm.rest_q)), f"the {side} arm's tool at rest")
    with contextlib.ExitStack() as stack:
        # The model and the cameras first: if either fails, no arm has been powered.
        policy = create_policy(args.policy)
        stack.callback(policy.close)
        cameras = {}
        for role, open_camera in workcell.cameras.items():
            cameras[role] = open_camera()
            stack.callback(cameras[role].close)
        warm_up(workcell, policy, cameras, kinematics, args.prompt, args.max_wait_s)
        robots = {}
        for side, arm in workcell.arms.items():
            print(f"powering the {side} arm", flush=True)
            robots[side] = arm.open()
            stack.callback(robots[side].close)
        drive(workcell, policy, robots, kinematics, cameras, args.prompt, args.seconds, args.max_wait_s, args.out, args.replan_s, blend_s)


def observe(
    policy: Policy,
    cameras: Mapping[str, Camera],
    arms: Mapping[str, Robot],
    kinematics: Mapping[str, Kinematics],
    prompt: str,
    recorder: MotionRecorder | None = None,
) -> dict[str, JointState]:
    """Hands the policy one camera tick with the arms' states and tool poses, and returns those states."""
    frames = {role: camera.read() for role, camera in cameras.items()}
    states = {side: arm.read() for side, arm in arms.items()}
    poses = {side: kinematics[side].fk(state.q) for side, state in states.items()}
    t = time.monotonic()
    policy.observe(frames, states, poses, prompt, t)
    if recorder is not None:
        recorder.offer(frames, t)
    return states


def warm_up(
    workcell: Workcell,
    policy: Policy,
    cameras: Mapping[str, Camera],
    kinematics: Mapping[str, Kinematics],
    prompt: str,
    max_wait_s: float,
) -> None:
    """Waits for the model's first chunk with every arm unpowered, and checks each tool is in its workspace."""
    arms = {side: arm.open_passive(workcell.rest_grip) for side, arm in workcell.arms.items()}
    stop = threading.Event()  # never set: nothing else is running to stop this
    try:
        for side, arm in arms.items():
            workcell.arms[side].safety.require_inside(kinematics[side].fk(arm.read().q), f"the {side} arm's tool")
        end = time.monotonic() + WINDOW_FILL_S
        while time.monotonic() < end:
            observe(policy, cameras, arms, kinematics, prompt)
        infer(policy, stop, max_wait_s, refill=lambda: observe(policy, cameras, arms, kinematics, prompt))
        print("the model is answering", flush=True)
    finally:
        for arm in arms.values():
            arm.close()
    policy.reset_episode()


def drive(
    workcell: Workcell,
    policy: Policy,
    robots: Mapping[str, Robot],
    kinematics: Mapping[str, Kinematics],
    cameras: Mapping[str, Camera],
    prompt: str,
    seconds: float,
    max_wait_s: float,
    out: Path,
    replan_s: float = REPLAN_S,
    blend_s: float = BLEND_S,
) -> None:
    """Runs the observer, the planner and the control loop, then returns the arms to rest."""
    trajectories = {side: Trajectory(blend_s) for side in robots}
    envelopes = {side: workcell.arms[side].safety for side in robots}
    stop = threading.Event()
    observed = threading.Event()
    ## plan_gaps: each time the plan ran out before the next chunk came, as [run time it ran out, seconds held].
    ## workspace_clamped_ticks, speed_limited_ticks: per side, the ticks the safety envelope moved a tool target into
    ## its workspace and slowed a joint.
    stats = {
        "policy_calls": 0,
        "chunks": 0,
        "stale_chunks": 0,
        "latency_ms": [],
        "plan_gaps": [],
        "workspace_clamped_ticks": {side: 0 for side in robots},
        "speed_limited_ticks": {side: 0 for side in robots},
    }
    log: dict[str, list] = {"t": [], "chunk_arrived": []}
    for side in robots:
        for name in ("commanded", "measured", "chunk_t", "chunk_poses", "chunk_grip"):
            log[f"{side}_{name}"] = []

    def watch() -> None:
        """Hands the policy every camera tick with the arms' measured state, and the recorder the frames."""
        while not stop.is_set():
            observe(policy, cameras, robots, kinematics, prompt, recorder)
            observed.set()

    def plan() -> None:
        """Asks for a chunk every `replan_s` and fades each into the trajectories."""
        while not observed.wait(timeout=0.1):
            if stop.is_set():
                return
        if stop.wait(WINDOW_FILL_S):
            return
        next_ask = time.monotonic()
        late_since: float | None = None  # when the first chunk too late to use arrived, while no plan is in force
        while not stop.wait(max(next_ask - time.monotonic(), 0.0)):
            next_ask = time.monotonic() + replan_s
            chunks = infer(policy, stop, max_wait_s, refill=None)
            if chunks is None:
                return
            now = time.monotonic()
            stats["policy_calls"] += 1
            # From the observation the chunk was planned from to its arrival.
            latency_ms = 1000.0 * (now - min(chunk.t[0] for chunk in chunks.values()))
            stats["latency_ms"].append(round(latency_ms, 1))
            if any(chunk.t[-1] <= now for chunk in chunks.values()):
                # Its last row was already due when it arrived. The plan in force stands; if it runs out, the
                # control loop stops the run.
                stats["stale_chunks"] += 1
                print(f"dropped a chunk that arrived after its last row was due ({latency_ms:.0f} ms)", flush=True)
                if not all(trajectory.has_plan() for trajectory in trajectories.values()):
                    # No plan is in force yet, so none can run out for the control loop to time: the run stops here.
                    if late_since is None:
                        late_since = now
                    elif now - late_since > PLAN_GAP_S:
                        raise RuntimeError(
                            f"no chunk has arrived in time to use and no plan is in force: every one for "
                            f"{now - late_since:.1f} s came after its last row was due (the last took "
                            f"{latency_ms:.0f} ms), longer than {PLAN_GAP_S:g} s"
                        )
                continue
            for side, chunk in chunks.items():
                if not trajectories[side].has_plan():
                    # The first chunk fades in from the arm standing where it is.
                    reading = robots[side].read()
                    trajectories[side].hold(kinematics[side].fk(reading.q), reading.grip, now)
                trajectories[side].update(chunk, now)
                log[f"{side}_chunk_t"].append(chunk.t)
                log[f"{side}_chunk_poses"].append(chunk.T_base_tcp)
                log[f"{side}_chunk_grip"].append(chunk.grip)
            log["chunk_arrived"].append(now)
            stats["chunks"] += 1

    started = time.monotonic()
    recorder = MotionRecorder(out / "video", tuple(cameras), started)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clank")
    workers = [pool.submit(watch), pool.submit(plan)]
    try:
        commanded = {side: robot.read() for side, robot in robots.items()}
        q = {side: reading.q.copy() for side, reading in commanded.items()}
        # The command filter's two stages, each a first-order low-pass, starting where the arms stand.
        stage1 = {side: reading.q.copy() for side, reading in commanded.items()}
        sent = {side: reading.q.copy() for side, reading in commanded.items()}
        grip = {side: reading.grip for side, reading in commanded.items()}
        trailing_since: dict[str, float | None] = {side: None for side in robots}
        first_side = next(iter(robots))
        gap_since: float | None = None  # when the plan in force ran out, while the arms hold its end
        dt = 1.0 / CONTROL_HZ
        next_tick, next_status, last_tick = started, started + 1.0, started - dt
        ends = None  # set when the first chunk is in force: the model's time starts then
        print("holding the arms where they are until the first chunk arrives", flush=True)
        while ends is None or time.monotonic() < ends:
            now = time.monotonic()
            since_last, last_tick = now - last_tick, now
            if ends is None and all(trajectory.has_plan() for trajectory in trajectories.values()):
                ends = now + seconds
                print(f"the model drives the arms for {seconds:.0f} s", flush=True)
            for worker in workers:
                if worker.done():
                    worker.result()  # raises what stopped it
                    raise RuntimeError("a worker thread ended while the run was going")
            tick = {"t": now - started}  # logged whole at the end of the tick, so a stop mid-tick leaves no ragged row
            readings = {}
            for side, robot in robots.items():
                reading = readings[side] = robot.read()
                trailing = float(np.abs(sent[side] - reading.q).max())
                if trailing <= FOLLOW_LIMIT_RAD:
                    trailing_since[side] = None
                elif trailing_since[side] is None:
                    trailing_since[side] = now
                elif now - trailing_since[side] > FOLLOW_PATIENCE_S:
                    raise RuntimeError(
                        f"the {side} arm has been {np.degrees(trailing):.0f} deg from its target for "
                        f"{now - trailing_since[side]:.1f} s: it is blocked or no longer following"
                    )
                if trajectories[side].has_plan():
                    target, grip[side], held = trajectories[side].sample_held(now)
                    if held > PLAN_GAP_S:
                        raise RuntimeError(
                            f"the model's plan ran out {held:.1f} s ago and no new chunk has come: "
                            f"waited longer than {PLAN_GAP_S:g} s"
                        )
                    if held > 0 and side == first_side:
                        # One gap per plan: the arms' plans come from the same chunks and run out together.
                        if gap_since is None:
                            gap_since = now - held
                            print("the plan ran out; holding the arms there until the next chunk", flush=True)
                    elif held == 0 and side == first_side and gap_since is not None:
                        stats["plan_gaps"].append([round(gap_since - started, 3), round(now - gap_since, 3)])
                        gap_since = None
                    target, clamped = envelopes[side].inside(target)
                    if clamped:
                        stats["workspace_clamped_ticks"][side] += 1
                    q_next = kinematics[side].ik_step(q[side], target, dt)
                    lead = np.abs(q_next - q[side])
                    if np.any(lead > PLAN_LEAD_LIMIT_RAD):
                        raise RuntimeError(
                            f"the IK wants the {side} arm's joints {np.degrees(lead).round(1).tolist()} deg from where "
                            f"its speed limits let them be, more than {np.degrees(PLAN_LEAD_LIMIT_RAD):.0f} deg: the plan "
                            f"is far faster than the limits allow, or the IK flipped to another solution"
                        )
                    # The step is shortened to the speed limits here, before the filter, so the IK solves the next
                    # tick from where the arm is told to be and keeps the tool on its way. The filter only averages
                    # these steps, so it never makes a joint faster.
                    q[side], slowed = envelopes[side].step(q[side], q_next, min(since_last, dt))
                    if slowed:
                        stats["speed_limited_ticks"][side] += 1
                # The filter runs on the time that passed, so a tick that comes early after a late one moves the
                # joints no faster: it would otherwise take a full tick's share in half the time.
                share = 1.0 - np.exp(-2.0 * np.pi * COMMAND_FILTER_HZ * since_last)
                stage1[side] += share * (q[side] - stage1[side])
                # The filter's share changes with the tick's length, which can carry a joint a hair past its limit
                # over one tick: what is sent is held to the limits as well, by the time that actually passed.
                sent[side], _ = envelopes[side].step(sent[side], sent[side] + share * (stage1[side] - sent[side]), since_last)
                envelopes[side].require_inside(
                    kinematics[side].fk(sent[side]), f"the {side} arm's commanded tool", WORKSPACE_SLACK_M
                )
                robot.command(sent[side], grip[side])
                tick[f"{side}_commanded"] = np.append(sent[side], grip[side])
                tick[f"{side}_measured"] = np.append(reading.q, reading.grip)
            for name, value in tick.items():
                log[name].append(value)
            recorder.note_arms(now, readings)
            if now >= next_status:
                next_status += 1.0
                trail = {
                    side: round(float(np.degrees(np.abs(tick[f"{side}_commanded"][:-1] - tick[f"{side}_measured"][:-1]).max())), 1)
                    for side in robots
                }
                print(
                    f"t={now - started:5.1f} s  chunks {stats['chunks']} (stale {stats['stale_chunks']})  "
                    f"furthest joint from target, deg {trail}",
                    flush=True,
                )
            next_tick += dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # a late tick: carry on from now rather than rushing to catch up
        if gap_since is not None:
            stats["plan_gaps"].append([round(gap_since - started, 3), round(time.monotonic() - gap_since, 3)])
    finally:
        stop.set()
        try:
            park(workcell, robots)
        finally:
            pool.shutdown(wait=True)
            for worker in workers:
                error = worker.exception()
                if error is not None:
                    traceback.print_exception(error)
            try:
                # The camera loop has stopped, so nothing more is offered: the videos can be finished.
                stats["video"] = recorder.close()
            finally:
                np.savez(out / "log.npz", started=started, **{name: np.array(values) for name, values in log.items()})
                (out / "stats.json").write_text(json.dumps({"prompt": prompt, **stats}, indent=2))
    for worker in workers:
        worker.result()


def infer(
    policy: Policy, stop: threading.Event, max_wait_s: float, refill: Callable[[], object] | None
) -> dict[str, PoseChunk] | None:
    """Asks for a chunk until the model is ready, calling `refill` near the end of each wait; None on stop."""
    asked = time.monotonic()
    while True:
        try:
            return policy.infer()
        except PolicyNotReady as error:
            wait, why = error.retry_after_s, error.reason
        if time.monotonic() - asked > max_wait_s:
            raise TimeoutError(f"the model was still not answering after {max_wait_s:.0f} s ({why})")
        print(f"model not ready ({why}); asking again in {wait:.0f} s", flush=True)
        if refill is None:
            if stop.wait(wait):
                return None
            continue
        if stop.wait(max(wait - WINDOW_FILL_S, 0.0)):
            return None
        end = time.monotonic() + min(wait, WINDOW_FILL_S)
        while time.monotonic() < end:
            refill()


def park(workcell: Workcell, robots: Mapping[str, Robot]) -> None:
    """Returns every arm to rest along a joint-space line, together, within each arm's speed limits."""
    start = {side: robot.read() for side, robot in robots.items()}
    rest = {side: np.array(workcell.arms[side].rest_q) for side in robots}
    park_s = max([PARK_S, *(workcell.arms[side].safety.travel_s(start[side].q, rest[side]) for side in robots)])
    print(f"returning the arms to rest over {park_s:.1f} s", flush=True)
    # How far along the line is read from the clock, not counted in steps, so a late or early step never makes one
    # move faster than the whole return does.
    began = time.monotonic()
    alpha = 0.0
    while alpha < 1.0:
        time.sleep(1.0 / PARK_HZ)
        alpha = min((time.monotonic() - began) / park_s, 1.0)
        for side, robot in robots.items():
            origin = start[side]
            robot.command(
                (1 - alpha) * origin.q + alpha * rest[side], origin.grip + alpha * (workcell.rest_grip - origin.grip)
            )
    time.sleep(0.5)


if __name__ == "__main__":
    main()
