"""Models by name: `create_policy("<provider id>/<model id>")` returns the model in clank's Policy slot.

A protocol (`clank/ai/protocols/`) is the code that talks to one kind of service and turns its answers into pose
chunks. A provider is a declaration: the service's name, the protocol that speaks to it, and each of its models with
the config that protocol needs.

A provider is a file (`clank/ai/providers/<provider id>.py`) or an installed distribution, which registers it under
the `clank.providers` entry point. The files come first, so a provider that is not in the checkout, such as a
service's own client, is installed beside clank and named the same way.
"""

from clank.ai import providers
from clank.ai.model import Model, Provider
from clank.ai.policy import Policy, PolicyNotReady
from clank.registry import declared_names, load_declared, load_installed

__all__ = ["Model", "Policy", "PolicyNotReady", "Provider", "create_policy", "find_model", "load_provider"]

PROVIDERS = "clank.providers"


def create_policy(model_ref: str) -> Policy:
    """Opens the model `"<provider id>/<model id>"` names and returns it, ready for observe() and infer()."""
    provider, model = find_model(model_ref)
    return provider.protocol(model.config)


def find_model(model_ref: str) -> tuple[Provider, Model]:
    """The declarations `"<provider id>/<model id>"` names: the provider and its model. Nothing is opened."""
    provider_id, slash, model_id = model_ref.partition("/")
    if not (provider_id and slash and model_id):
        raise ValueError(f"{model_ref!r} does not name a model: a model is named '<provider id>/<model id>'")
    provider = load_provider(provider_id)
    for model in provider.models:
        if model.id == model_id:
            return provider, model
    raise LookupError(
        f"provider {provider.id!r} ({provider.name}) has no model {model_id!r}; "
        f"its models are {[model.id for model in provider.models]}"
    )


def load_provider(provider_id: str) -> Provider:
    """Imports `clank.ai.providers.<provider_id>`, or loads the provider an installed distribution registers as
    `provider_id` under `clank.providers`.

    A provider is imported only when it is asked for, so a checkout without that provider's file, or a machine
    without the client library its protocol needs, still imports `clank.ai` and every other provider.
    """
    present = declared_names(providers)
    if provider_id in present:
        return load_declared(providers, provider_id, "provider")
    installed = load_installed(PROVIDERS, provider_id, "provider")
    if installed is not None:
        return installed
    raise LookupError(
        f"there is no provider {provider_id!r}; clank.ai.providers has {present}, and no installed distribution "
        f"registers one under {PROVIDERS}"
    )
