"""nerve: a language model driving arms through a small vocabulary of moves.

Three parts, each replaceable on its own:

- `primitives.py`: the vocabulary. Each primitive is one function from where the arms are to where they should go.
- `agent.py`: the loop. One model turn per chunk: show the model the scene, run the primitives it calls, hand clank
  the moves.
- a wire file per kind of service (`openai_chat.py`, `openai_responses.py`, `anthropic_messages.py`): how the
  conversation reaches a model.

A provider file names one of the functions below as its protocol, with the service's address, key and model in each
model's config:

    provider = Provider(
        id="...",
        name="...",
        protocol=nerve.openai_chat,
        models=(
            Model(id="...", config={"base_url": "...", "model": "...", "api_key_env": "...", "timeout_s": 60, "think_s": 90}),
        ),
    )
"""

from collections.abc import Mapping
from typing import Any

from clank.ai.protocols.nerve.agent import NervePolicy
from clank.ai.protocols.nerve.anthropic_messages import AnthropicMessages
from clank.ai.protocols.nerve.openai_chat import OpenAIChat
from clank.ai.protocols.nerve.openai_responses import OpenAIResponses

__all__ = ["NervePolicy", "anthropic_messages", "openai_chat", "openai_responses"]


def openai_chat(config: Mapping[str, Any]) -> NervePolicy:
    """The protocol for a service that speaks OpenAI's chat-completions format."""
    return NervePolicy(config, OpenAIChat(config))


def openai_responses(config: Mapping[str, Any]) -> NervePolicy:
    """The protocol for OpenAI's Responses API, which offers service tiers such as Ultrafast."""
    return NervePolicy(config, OpenAIResponses(config))


def anthropic_messages(config: Mapping[str, Any]) -> NervePolicy:
    """The protocol for Anthropic's Messages API. It needs the `anthropic` extra."""
    return NervePolicy(config, AnthropicMessages(config))
