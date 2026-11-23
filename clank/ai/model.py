"""What a provider file declares: the provider and the models it serves."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from clank.ai.policy import Policy


@dataclass(frozen=True)
class Model:
    """One model a provider serves.

    id: the model's name within its provider: the part after the slash in `"<provider id>/<model id>"`.
    config: what the provider's protocol needs to open this model, in that protocol's own terms.
    blend_s: how long each new chunk takes to fade in from the plan in force, seconds; None for the run's own.
    """

    id: str
    config: Mapping[str, Any]
    blend_s: float | None = None


@dataclass(frozen=True)
class Provider:
    """A service that hosts models.

    id: the provider's name, which is also its file's: `clank/ai/providers/<id>.py`.
    name: what to call the provider in a message to a person.
    protocol: the code that speaks to the provider: called with one model's `config`, it opens that model and
        returns it as a Policy.
    models: the models the provider serves.
    """

    id: str
    name: str
    protocol: Callable[[Mapping[str, Any]], Policy]
    models: tuple[Model, ...]
