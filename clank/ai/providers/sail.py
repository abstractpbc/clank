"""Sail Research: image-reading open-weight models through nerve, keyed by `SAIL_API_KEY`."""

from clank.ai.model import Model, Provider
from clank.ai.protocols import nerve

_COMMON = {
    "base_url": "https://api.sailresearch.com/v1",
    "api_key_env": "SAIL_API_KEY",
    "timeout_s": 60,
    "think_s": 90,
}

provider = Provider(
    id="sail",
    name="Sail Research",
    protocol=nerve.openai_chat,
    models=tuple(
        Model(id=model_id, config={**_COMMON, "model": model_id})
        for model_id in (
            "moonshotai/Kimi-K3.1-Pro",
            "zai-org/GLM-5.3-Flash",
            "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "google/gemma-4-31B-it",
            "nvidia/Gemma-4-31B-IT-NVFP4",
        )
    ),
)
