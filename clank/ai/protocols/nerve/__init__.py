"""nerve: a language model driving arms through primitives, an agent loop, and one wire file per service."""

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
