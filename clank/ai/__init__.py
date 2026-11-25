"""Models by name: `create_policy("<provider id>/<model id>")` returns a provider file's or entry point's model."""

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
    """The provider `clank.ai.providers.<provider_id>` declares, else the one installed under `clank.providers`."""
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
