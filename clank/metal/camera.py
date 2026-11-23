"""The slot one camera's driver plugs into."""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Camera(Protocol):
    def read(self) -> npt.NDArray[np.uint8]:
        """The next frame, (H, W, 3) BGR; waits for it."""
        ...

    def close(self) -> None: ...
