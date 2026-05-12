"""Quanto adjustment for risk-neutral pricing under FCA.

When pricing in a foreign measure, Brownian shocks must be adjusted
for the correlation with FX rates.  The quanto adjustment is:

    z_adj = z - Σᵢ ρᵢ · σ_fx_i · √dt

where ρᵢ is the correlation between the model's shock and FX shock i,
and σ_fx_i is the volatility of FX rate i.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def quanto_adjustment(
    shock: Array,
    correlations: Array,
    fx_vols: Array,
    dt: float,
) -> Array:
    """Apply quanto adjustment to a Brownian shock.

    Adjusts a shock for the correlation with one or more FX rates,
    which is required when pricing in a foreign risk-neutral measure.

    The adjustment is:
        z_adj = z - Σᵢ ρ(model, fx_i) · σ_fx_i · √dt

    Args:
        shock: Original N(0,1) shock (scalar JAX array).
        correlations: Array of correlations ρ(model, fx_i) for each
            FX rate, shape ``(n_fx,)``.
        fx_vols: Array of FX volatilities σ_fx_i, shape ``(n_fx,)``.
        dt: Timestep size in years.

    Returns:
        Quanto-adjusted shock (scalar JAX array).
    """
    correlations = jnp.asarray(correlations, dtype=jnp.float64)
    fx_vols = jnp.asarray(fx_vols, dtype=jnp.float64)
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    adjustment = jnp.sum(correlations * fx_vols) * jnp.sqrt(dt_arr)
    return shock - adjustment


def quanto_drift_adjustment(
    correlations: Array,
    fx_vols: Array,
    model_vol: float,
) -> Array:
    """Compute the quanto drift adjustment term.

    The drift adjustment in the quanto measure is:
        drift_adj = -σ_model · Σᵢ ρᵢ · σ_fx_i

    This is added to the drift of the foreign-currency SDE when
    pricing in the domestic measure.

    Args:
        correlations: Array of correlations ρ(model, fx_i), shape ``(n_fx,)``.
        fx_vols: Array of FX volatilities σ_fx_i, shape ``(n_fx,)``.
        model_vol: Volatility of the model being adjusted.

    Returns:
        Quanto drift adjustment (scalar JAX array).
    """
    correlations = jnp.asarray(correlations, dtype=jnp.float64)
    fx_vols = jnp.asarray(fx_vols, dtype=jnp.float64)
    return -jnp.asarray(model_vol, dtype=jnp.float64) * jnp.sum(
        correlations * fx_vols
    )
