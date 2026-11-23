"""The wire for a service that speaks OpenAI's chat-completions format: `POST <base_url>/chat/completions`, with
images as data URLs and primitives as function tools. Many services other than OpenAI accept this format."""

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from clank.ai.policy import PolicyNotReady
from clank.ai.protocols.common import require_env
from clank.ai.protocols.nerve.llm import Call, Tool, Turn


class OpenAIChat:
    def __init__(self, config: Mapping[str, Any]) -> None:
        """
        config:
            base_url: the service's API root, e.g. `https://api.openai.com/v1`.
            model: the model's name at that service.
            api_key_env: the environment variable holding the key.
            timeout_s: how long one answer may take.
        """
        self._url = f"{str(config['base_url']).rstrip('/')}/chat/completions"
        self._model = str(config["model"])
        self._timeout_s = float(config["timeout_s"])
        self._key = require_env(str(config["api_key_env"]), f"the key for {self._url}")

    def reply(self, turns: Sequence[Turn], tools: Sequence[Tool]) -> Turn:
        body = {
            "model": self._model,
            "messages": [_message(turn) for turn in turns],
            "tools": [
                {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}
                for tool in tools
            ],
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            wait = error.headers.get("Retry-After")
            if error.code in (429, 503) and wait is not None and wait.strip().isdigit():
                # The service is busy and says when to come back.
                raise PolicyNotReady(f"{self._url} answered {error.code}", float(wait)) from error
            raise RuntimeError(f"{self._url} answered {error.code}: {error.read().decode(errors='replace')[:500]}") from error
        message = payload["choices"][0]["message"]
        return Turn(
            "assistant",
            message.get("content") or "",
            calls=tuple(
                Call(id=call["id"], name=call["function"]["name"], arguments=_arguments(call["function"]))
                for call in message.get("tool_calls") or ()
            ),
        )

    def close(self) -> None:
        pass


def _message(turn: Turn) -> dict[str, Any]:
    """One turn in the chat-completions format."""
    if turn.role == "user":
        content: list[dict[str, Any]] = [{"type": "text", "text": turn.text}]
        for image in turn.images:
            url = f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}
    if turn.role == "assistant":
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if turn.calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in turn.calls
            ]
        return message
    if turn.role == "tool":
        return {"role": "tool", "tool_call_id": turn.call_id, "content": turn.text}
    return {"role": "system", "content": turn.text}


def _arguments(function: Mapping[str, Any]) -> Mapping[str, Any]:
    """A call's arguments, which the format carries as a JSON string."""
    arguments = json.loads(function["arguments"])
    if not isinstance(arguments, dict):
        raise ValueError(f"the model called {function['name']} with arguments that are not an object: {arguments!r}")
    return arguments
