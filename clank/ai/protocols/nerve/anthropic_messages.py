"""The wire for Anthropic's Messages API, through Anthropic's own client library: images as base64 blocks,
primitives as tools, and each call's result as a `tool_result` block.

The Messages format differs from the conversation the loop keeps in three ways, each bridged here:

- The system turn is the request's `system`, not a message.
- A call's result is a block of a user message. The tool turns that follow an assistant turn, and the observation
  after them, go out as one user message with the results first.
- The model's thinking is not sent back. A `Turn` keeps a turn's words and calls only, and the loop takes the images
  out of earlier observations on every request, which would void thinking blocks that are bound to the history they
  were written in. The model reasons anew each turn, from the conversation's words, calls and results.
"""

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from clank.ai.policy import PolicyNotReady
from clank.ai.protocols.nerve.llm import Call, Tool, Turn

_FALLBACK_BETA = "server-side-fallback-2026-07-01"

class AnthropicMessages:
    def __init__(self, config: Mapping[str, Any]) -> None:
        """
        config:
            model: the model's id at Anthropic.
            timeout_s: how long one answer may take.
            max_tokens: the most the model may write in one answer, its thinking included.
            effort: how hard the model thinks: `low`, `medium`, `high`, `xhigh` or `max`.

        The key is the one Anthropic's library finds: `ANTHROPIC_API_KEY`, or the profile `ant auth login` saved. A
        missing one is raised by the first reply(), not here. The library is imported here, when a model is opened,
        so that nerve's other wires need only their own.
        """
        import anthropic

        self._sdk = anthropic
        self._model = str(config["model"])
        self._max_tokens = int(config["max_tokens"])
        self._effort = str(config["effort"])
        # One try per ask: clank's loop waits and asks again itself, and a retry here would outlast `timeout_s`.
        self._client = anthropic.Anthropic(timeout=float(config["timeout_s"]), max_retries=0)

    def reply(self, turns: Sequence[Turn], tools: Sequence[Tool]) -> Turn:
        system, messages = _messages(turns)
        try:
            # A request the model declines is answered by the model Anthropic routes that kind of refusal to, within
            # the same call; a `refusal` stop reason then means every model declined.
            message = self._client.beta.messages.create(
                betas=[_FALLBACK_BETA],
                fallbacks="default",
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                tools=[
                    {"name": tool.name, "description": tool.description, "input_schema": dict(tool.parameters)}
                    for tool in tools
                ],
                output_config={"effort": self._effort},
            )
        except (self._sdk.RateLimitError, self._sdk.InternalServerError) as error:
            wait = error.response.headers.get("retry-after")
            if error.status_code in (429, 503, 529) and wait is not None and wait.strip().isdigit():
                # The service is busy and says when to come back.
                raise PolicyNotReady(f"Anthropic answered {error.status_code}", float(wait)) from error
            raise
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details is not None else None
            raise RuntimeError(f"{self._model} declined to answer (category: {category})")
        if message.stop_reason == "max_tokens":
            raise RuntimeError(
                f"{self._model}'s answer was cut off at max_tokens ({self._max_tokens}), so a call in it may be incomplete"
            )
        return Turn(
            "assistant",
            "".join(block.text for block in message.content if block.type == "text"),
            calls=tuple(
                Call(id=block.id, name=block.name, arguments=block.input)
                for block in message.content
                if block.type == "tool_use"
            ),
        )

    def close(self) -> None:
        self._client.close()


def _messages(turns: Sequence[Turn]) -> tuple[str, list[dict[str, Any]]]:
    """The conversation in the Messages format: the system turns' words, and the rest as messages. Turns that follow
    one another on the same side go out as one message, and an assistant turn with neither words nor calls, which
    the format has no message for, is left out."""
    system = "\n\n".join(turn.text for turn in turns if turn.role == "system")
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if turn.role == "system":
            continue
        role = "assistant" if turn.role == "assistant" else "user"
        blocks = _blocks(turn)
        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    return system, messages


def _blocks(turn: Turn) -> list[dict[str, Any]]:
    """One turn's content blocks."""
    if turn.role == "tool":
        return [{"type": "tool_result", "tool_use_id": turn.call_id, "content": turn.text}]
    blocks: list[dict[str, Any]] = [{"type": "text", "text": turn.text}] if turn.text else []
    for image in turn.images:
        source = {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}
        blocks.append({"type": "image", "source": source})
    for call in turn.calls:
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)})
    return blocks
