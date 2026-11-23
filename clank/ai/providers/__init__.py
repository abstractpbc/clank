"""Provider declarations, one file per provider in the checkout: `<provider id>.py`, holding a `provider`.

Nothing is imported here: `clank.ai.load_provider()` imports a provider's file when that provider is asked for, and
falls back on an installed distribution that registers the same name under `clank.providers`.
"""
