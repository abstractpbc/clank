"""Helpers every protocol may use."""

import os


def require_env(name: str, holds: str) -> str:
    """The environment variable `name`, stripped; RuntimeError saying what it `holds` when unset or blank."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"the environment variable {name} is not set: it holds {holds}")
    return value
