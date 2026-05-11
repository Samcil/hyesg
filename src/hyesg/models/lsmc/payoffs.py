"""Standard payoff functions for LSMC pricing."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def european_put(spot: Array, strike: float) -> Array:
    """European put payoff max(K - S, 0).

    Args:
        spot: Spot prices, shape (n_paths,).
        strike: Strike price.

    Returns:
        Payoff values, shape (n_paths,).
    """
    return jnp.maximum(strike - spot, 0.0)


def american_put(spot: Array, strike: float) -> Array:
    """American put exercise value max(K - S, 0).

    Identical to the European put at any single exercise date;
    the early-exercise logic is handled by the LSMC backward pass.

    Args:
        spot: Spot prices, shape (n_paths,).
        strike: Strike price.

    Returns:
        Exercise values, shape (n_paths,).
    """
    return jnp.maximum(strike - spot, 0.0)


def bermudan_put(spot: Array, strike: float) -> Array:
    """Bermudan put exercise value max(K - S, 0).

    Identical payoff to European/American at any exercise date;
    the restricted exercise schedule is handled by the LSMC pricer.

    Args:
        spot: Spot prices, shape (n_paths,).
        strike: Strike price.

    Returns:
        Exercise values, shape (n_paths,).
    """
    return jnp.maximum(strike - spot, 0.0)
