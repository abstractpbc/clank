"""Helpers every protocol may use."""

import os


def require_env(name: str, holds: str) -> str:
    """The environment variable `name`, surrounding whitespace removed. RuntimeError when it is unset or blank, saying
    what it should hold."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"the environment variable {name} is not set: it holds {holds}")
    return value
