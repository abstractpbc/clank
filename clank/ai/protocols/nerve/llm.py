"""What the agent loop needs from a language model: conversation and tools in, one assistant turn out."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Tool:
    """Something the model may call; `parameters` is the JSON Schema of its arguments."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class Call:
    """One call the model made. `id` ties the call to the tool turn that answers it."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Turn:
    """One conversation entry: system, user (with images), assistant (with calls), or tool (with call_id)."""

    role: Literal["system", "user", "assistant", "tool"]
    text: str
    images: tuple[bytes, ...] = ()
    calls: tuple[Call, ...] = ()
    call_id: str | None = None


class LLM(Protocol):
    def reply(self, turns: Sequence[Turn], tools: Sequence[Tool]) -> Turn:
        """The model's next assistant turn; PolicyNotReady when the service says to come back later."""
        ...

    def close(self) -> None: ...
