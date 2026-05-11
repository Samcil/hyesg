"""Model registry for hyesg.

Replaces C#'s MEF-based model discovery with an explicit,
type-safe registry.
"""

from __future__ import annotations

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class.

    Args:
        name: Unique registry key for the model.

    Returns:
        Decorator that registers the class.

    Raises:
        ValueError: If name is already registered.
    """

    def decorator(cls: type) -> type:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type:
    """Retrieve a registered model class by name.

    Args:
        name: Registry key.

    Returns:
        The model class.

    Raises:
        KeyError: If model not found.
    """
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
