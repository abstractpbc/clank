"""Fake: each `runs/<name>/log.npz` in the working directory played back as the model `fake/<name>`."""

from pathlib import Path

from clank.ai.model import Model, Provider
from clank.ai.protocols.recording import RecordingPolicy

_RUNS = Path("runs")

provider = Provider(
    id="fake",
    name="Fake (recorded runs)",
    protocol=RecordingPolicy,
    models=tuple(
        Model(id=log.parent.name, config={"recording": str(log)}) for log in sorted(_RUNS.glob("*/log.npz"))
    ),
)
