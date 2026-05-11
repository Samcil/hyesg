"""Model registry for hyesg.

Replaces C#'s MEF-based model discovery with an explicit,
type-safe registry.
"""

from __future__ import annotations

_MODEL_REGISTRY: dict[str, type] = {}
_REGISTERED_MODULES: set[str] = set()


def register_model(name: str):
    """Decorator to register a model class.

    Idempotent: re-registering the same class under the same name
    is a no-op.  Registering a *different* class under an existing
    name raises ``ValueError``.

    Args:
        name: Unique registry key for the model.

    Returns:
        Decorator that registers the class.

    Raises:
        ValueError: If name is already registered with a different class.
    """

    def decorator(cls: type) -> type:
        if name in _MODEL_REGISTRY:
            if _MODEL_REGISTRY[name] is cls:
                return cls  # idempotent
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        _REGISTERED_MODULES.add(cls.__module__)
        return cls

    return decorator


def _ensure_populated() -> None:
    """Lazily populate registry if it was cleared (e.g. by test teardown).

    Only reloads modules that previously registered a model via
    ``@register_model``, avoiding side-effects on unrelated modules
    (e.g. dataclass re-creation breaking ``isinstance`` checks).
    """
    if _MODEL_REGISTRY:
        return
    import importlib
    import sys

    # Only reload modules that actually registered models
    for mod_name in list(_REGISTERED_MODULES):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    # If nothing was in sys.modules yet, import the models package
    if not _MODEL_REGISTRY:
        import hyesg.models  # noqa: F401


def get_model(name: str) -> type:
    """Retrieve a registered model class by name.

    Lazily ensures that models are registered before lookup,
    so callers never see an empty registry.

    Args:
        name: Registry key.

    Returns:
        The model class.

    Raises:
        KeyError: If model not found.
    """
    if name not in _MODEL_REGISTRY:
        _ensure_populated()
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model: '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered models. Useful for testing."""
    _MODEL_REGISTRY.clear()
