"""What the agent loop needs from a language model: the conversation and the tools go in, one assistant turn comes
out. A wire file (`openai_chat.py`, `openai_responses.py`, `anthropic_messages.py`) does that for one kind of service; the loop never sees the service's format."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Tool:
    """Something the model may call.

    parameters: the JSON Schema of the call's arguments object.
    """

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
    """One entry of the conversation.

    role: who is speaking. `system` sets the model up, `user` is an observation, `assistant` is the model, and
        `tool` is the result of one of the model's calls.
    text: the words.
    images: JPEG images, on a `user` turn.
    calls: the calls the model made, on an `assistant` turn.
    call_id: the call a `tool` turn answers.
    """

    role: Literal["system", "user", "assistant", "tool"]
    text: str
    images: tuple[bytes, ...] = ()
    calls: tuple[Call, ...] = ()
    call_id: str | None = None


class LLM(Protocol):
    def reply(self, turns: Sequence[Turn], tools: Sequence[Tool]) -> Turn:
        """The model's next turn (role `assistant`), given the conversation so far and what it may call.

        Raises `clank.ai.PolicyNotReady` when the service says to come back later.
        """
        ...

    def close(self) -> None: ...
