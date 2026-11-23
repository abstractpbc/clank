"""Anthropic (anthropic.com): Claude Fable 5.2 driving the arms through nerve's moves.

`anthropic/claude-fable-5-2` is the model. It needs the `anthropic` extra. The key is read from `ANTHROPIC_API_KEY`,
or from the profile `ant auth login` saved. Fable 5.2 requires an organization with 30-day data retention.
"""

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
