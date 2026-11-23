"""OpenAI (openai.com): GPT-6.2 Astra on the Ultrafast tier, driving the arms through nerve's moves.

`openai/gpt-6.2-astra` is the model. The key is read from `OPENAI_API_KEY`. Ultrafast is OpenAI's fastest service
tier, offered on the Responses API only. It has rate limits of its own, keeps GPT-6.2 Astra's data in the US, and
costs several times the standard rate.
"""

from clank.ai.model import Model, Provider
from clank.ai.protocols import nerve

provider = Provider(
    id="openai",
    name="OpenAI",
    protocol=nerve.openai_responses,
    models=(
        Model(
            id="gpt-6.2-astra",
            config={
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-6.2-astra",
                "api_key_env": "OPENAI_API_KEY",
                "service_tier": "ultrafast",
                "timeout_s": 60,
                "think_s": 90,
            },
        ),
    ),
)
