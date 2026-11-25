"""What a provider file declares: the provider and the models it serves."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from clank.ai.policy import Policy


@dataclass(frozen=True)
class Model:
    """One model a provider serves: its id, its protocol config, and its blend_s (None for the run's own)."""

    id: str
    config: Mapping[str, Any]
    blend_s: float | None = None


@dataclass(frozen=True)
class Provider:
    """A service that hosts models: its id (its file's name), display name, protocol, and models."""

    id: str
    name: str
    protocol: Callable[[Mapping[str, Any]], Policy]
    models: tuple[Model, ...]
