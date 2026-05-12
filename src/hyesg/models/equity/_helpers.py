"""Shared helpers for equity models."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from hyesg.outputs import OutputName


def extract_short_rate(deps: dict[str, Any]) -> jnp.ndarray:
    """Extract the short rate from nested dependency outputs.

    Scans dependency outputs for the first dict containing a short rate
    entry. Returns 0.0 if no short rate is found.

    Args:
        deps: Model dependency outputs dict.

    Returns:
        Short rate scalar array (float64).
    """
    for dep_out in deps.values():
        if isinstance(dep_out, dict) and OutputName.SHORT_RATE in dep_out:
            return dep_out[OutputName.SHORT_RATE]
    return jnp.array(0.0, dtype=jnp.float64)
