"""Anthropic: Claude Fable 5.2 through nerve; needs the `anthropic` extra and 30-day data retention."""

from clank.ai.model import Model, Provider
from clank.ai.protocols import nerve

provider = Provider(
    id="anthropic",
    name="Anthropic",
    protocol=nerve.anthropic_messages,
    models=(
        Model(
            id="claude-fable-5-2",
            config={
                "model": "claude-fable-5-2",
                "timeout_s": 90,
                "think_s": 120,
                "max_tokens": 16000,
                "effort": "high",
            },
        ),
    ),
)
