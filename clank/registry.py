"""Declarations by name: a package holding one module per declaration, each module named after what it declares."""

import importlib
import importlib.metadata
import pkgutil
from types import ModuleType
from typing import Any


def declared_names(package: ModuleType) -> list[str]:
    """The names a package of declarations holds, sorted: one module a name."""
    return sorted(module.name for module in pkgutil.iter_modules(package.__path__))


def load_declared(package: ModuleType, name: str, attribute: str) -> Any:
    """Imports `<package>.<name>` and returns its `attribute`, whose `id` must be `name`."""
    present = declared_names(package)
    if name not in present:
        raise LookupError(f"there is no {attribute} {name!r}; the {attribute}s in {package.__name__} are {present}")
    declared = getattr(importlib.import_module(f"{package.__name__}.{name}"), attribute)
    if declared.id != name:
        raise ValueError(f"{package.__name__}.{name} declares {attribute} {declared.id!r}, not {name!r}")
    return declared


def load_installed(group: str, name: str, attribute: str) -> Any | None:
    """The declaration an installed distribution registers under entry-point `group` as `name`, or None."""
    for entry in importlib.metadata.entry_points(group=group):
        if entry.name != name:
            continue
        declared = entry.load()
        found = getattr(declared, "id", name)
        if found != name:
            raise ValueError(f"the {group} entry point {name!r} declares {attribute} {found!r}, not {name!r}")
        return declared
    return None
