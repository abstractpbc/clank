"""One i2rt YAM arm read over CAN with every motor left off; it reports the aperture it is given."""

import time

import can
import numpy as np
import numpy.typing as npt

from clank.types import JointState

## CAN ids of the six arm joints, base to wrist; the gripper motor is 7 (i2rt's `yam_v1.yml`).
ARM_MOTOR_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_MOTOR_ID = 7
## Where a state query goes, and how far above its id a motor answers from.
_QUERY_ID = 0x7FF
_REPLY_OFFSET = 0x10
## A DM4310's and a DM4340's position field spans -12.5 to 12.5 rad in 16 bits.
_POSITION_MAX_RAD = 12.5
## The state frame's status nibble. 0 is disabled and 1 enabled. 0xD is "loss of communication": the motor switched
## itself off when its last controller stopped talking, which is how a parked arm is left; its angle still reads
## true. Anything else (voltage, current, temperature, overload) is a fault and read() refuses.
_STATUS_OK = (0x0, 0x1, 0xD)


class PassiveYam:
    def __init__(self, channel: str, grip: float, reply_timeout_s: float = 0.05) -> None:
        """channel is the SocketCAN interface, grip the aperture to report, reply_timeout_s each motor's wait."""
        if not 0.0 <= grip <= 1.0:
            raise ValueError(f"grip must be between 0 and 1, got {grip}")
        self._channel = channel
        self._grip = grip
        self._timeout = reply_timeout_s
        self._bus = can.Bus(channel=channel, interface="socketcan")

    def read(self) -> JointState:
        q = np.array([_wrap(self.motor_angle(motor_id)) for motor_id in ARM_MOTOR_IDS])
        return JointState(t=time.monotonic(), q=q, grip=self._grip)

    def motor_angle(self, motor_id: int) -> float:
        """One motor's angle as the motor reports it, radians."""
        self._bus.send(
            can.Message(arbitration_id=_QUERY_ID, data=[motor_id, 0x00, 0xCC, 0, 0, 0, 0, 0], is_extended_id=False)
        )
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            reply = self._bus.recv(timeout=remaining) if remaining > 0 else None
            if reply is None:
                raise TimeoutError(f"{self._channel}: motor {motor_id} did not answer within {self._timeout} s")
            if reply.arbitration_id == motor_id + _REPLY_OFFSET and len(reply.data) == 8:
                break
        status = reply.data[0] >> 4
        if status not in _STATUS_OK:
            raise RuntimeError(f"{self._channel}: motor {motor_id} reports fault status {status:#x}")
        raw = (reply.data[1] << 8) | reply.data[2]
        return raw / 0xFFFF * 2 * _POSITION_MAX_RAD - _POSITION_MAX_RAD

    def command(self, q: npt.NDArray[np.float64], grip: float) -> None:
        raise RuntimeError("PassiveYam only reads; it never sends a target")

    def close(self) -> None:
        self._bus.shutdown()


def _wrap(angle: float) -> float:
    """Brings a motor angle a whole turn off into (-pi, pi], as i2rt does when it connects."""
    if angle < -np.pi:
        return angle + 2 * np.pi
    if angle > np.pi:
        return angle - 2 * np.pi
    return angle
