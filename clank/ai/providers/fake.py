"""Fake: recorded runs played back as models. No service, no key, no model call.

Each live run under `runs/` that saved a `log.npz` is a model, named after its directory: `fake/live-b` plays
`runs/live-b/log.npz`. `runs/` is looked for in the working directory, where the run scripts write it, once, when this
file is first imported.
"""

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
