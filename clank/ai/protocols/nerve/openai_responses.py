"""The wire for OpenAI's Responses API: `POST <base_url>/responses`, with images as data URLs, primitives as function
tools, and the service tier asked for on every request. Ultrafast is offered here and not on chat completions.

Nothing is stored at OpenAI (`store` is false): the whole conversation goes out each turn, as the loop keeps it. A
reasoning model's reasoning must go back with the results of the calls it made, so each answer's reasoning items
(their encrypted content included) are kept here, by the first call they led to, and sent back in front of those
calls. Reasoning that led to no call is not sent back.
"""

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from clank.ai.policy import PolicyNotReady
from clank.ai.protocols.common import require_env
from clank.ai.protocols.nerve.llm import Call, Tool, Turn


class OpenAIResponses:
    def __init__(self, config: Mapping[str, Any]) -> None:
        """
        config:
            base_url: the API root, `https://api.openai.com/v1`.
            model: the model's name at OpenAI.
            api_key_env: the environment variable holding the key.
            timeout_s: how long one answer may take.
            service_tier: the tier to be served at, e.g. `ultrafast`.
        """
        self._url = f"{str(config['base_url']).rstrip('/')}/responses"
        self._model = str(config["model"])
        self._timeout_s = float(config["timeout_s"])
        self._tier = str(config["service_tier"])
        self._key = require_env(str(config["api_key_env"]), f"the key for {self._url}")
        ## The reasoning items of each answer that made calls, keyed by the id of its first call.
        self._reasoning: dict[str, list[Mapping[str, Any]]] = {}
        self._told_tier = False

    def reply(self, turns: Sequence[Turn], tools: Sequence[Tool]) -> Turn:
        body = {
            "model": self._model,
            "instructions": "\n\n".join(turn.text for turn in turns if turn.role == "system"),
            "input": [item for turn in turns if turn.role != "system" for item in self._items(turn)],
            "tools": [
                {"type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters}
                for tool in tools
            ],
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "service_tier": self._tier,
        }
        payload = self._post(body)
        if payload["status"] != "completed":
            raise RuntimeError(f"{self._model}'s answer is {payload['status']}: {payload.get('incomplete_details')}")
        served = payload.get("service_tier")
        if served != self._tier and not self._told_tier:
            self._told_tier = True
            print(f"{self._model} was asked for the {self._tier} tier and answered at {served!r}", flush=True)
        text, calls, reasoning = [], [], []
        for item in payload["output"]:
            if item["type"] == "reasoning":
                reasoning.append(item)
            elif item["type"] == "function_call":
                calls.append(Call(id=item["call_id"], name=item["name"], arguments=_arguments(item)))
            elif item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "refusal":
                        raise RuntimeError(f"{self._model} declined to answer: {part['refusal']}")
                    if part["type"] == "output_text":
                        text.append(part["text"])
        if calls and reasoning:
            self._reasoning[calls[0].id] = reasoning
        return Turn("assistant", "".join(text), calls=tuple(calls))

    def close(self) -> None:
        self._reasoning.clear()

    def _items(self, turn: Turn) -> list[Mapping[str, Any]]:
        """One turn as Responses input items."""
        if turn.role == "user":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": turn.text}]
            for image in turn.images:
                url = f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
                content.append({"type": "input_image", "image_url": url})
            return [{"role": "user", "content": content}]
        if turn.role == "tool":
            return [{"type": "function_call_output", "call_id": turn.call_id, "output": turn.text}]
        items: list[Mapping[str, Any]] = []
        if turn.calls:
            items.extend(self._reasoning.get(turn.calls[0].id, ()))
        if turn.text:
            items.append({"role": "assistant", "content": turn.text})
        for call in turn.calls:
            items.append(
                {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": json.dumps(call.arguments)}
            )
        return items

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            wait = error.headers.get("Retry-After")
            if error.code in (429, 503) and wait is not None and wait.strip().isdigit():
                # The service is busy and says when to come back.
                raise PolicyNotReady(f"{self._url} answered {error.code}", float(wait)) from error
            raise RuntimeError(f"{self._url} answered {error.code}: {error.read().decode(errors='replace')[:500]}") from error


def _arguments(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """A call's arguments, which the format carries as a JSON string."""
    arguments = json.loads(item["arguments"])
    if not isinstance(arguments, dict):
        raise ValueError(f"the model called {item['name']} with arguments that are not an object: {arguments!r}")
    return arguments
