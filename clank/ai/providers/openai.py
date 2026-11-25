"""OpenAI: GPT-6.2 Astra on the Ultrafast tier through nerve, keyed by `OPENAI_API_KEY`."""

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
