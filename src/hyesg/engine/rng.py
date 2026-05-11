"""RNG subsystem for deterministic shock generation.

Provides a key hierarchy and vectorised shock generation using JAX's
functional PRNG.  All functions are pure and compatible with ``jax.jit``.

Key hierarchy::

    master_key
    ├── regime_0
    │   ├── trial_0
    │   ├── trial_1
    │   └── ...
    ├── regime_1
    │   └── ...
    └── ...
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Key hierarchy
# ---------------------------------------------------------------------------


def create_rng_keys(seed: int, n_trials: int, n_regimes: int) -> Array:
    """Generate deterministic, reproducible RNG keys.

    Builds a two-level hierarchy from a single integer seed:
    *master → regime → trial*.

    Args:
        seed: Integer seed for the master PRNG key.
        n_trials: Number of independent trials per regime.
        n_regimes: Number of regimes.

    Returns:
        Array of shape ``(n_regimes, n_trials, 2)`` containing one
        JAX PRNGKey per regime–trial combination.
    """
    master_key = jax.random.PRNGKey(seed)
    regime_keys = jax.random.split(master_key, n_regimes)
    # vmap over regimes: split each regime key into n_trials trial keys
    trial_keys = jax.vmap(lambda rk: jax.random.split(rk, n_trials))(regime_keys)
    return trial_keys


# ---------------------------------------------------------------------------
# Shock generation
# ---------------------------------------------------------------------------


def generate_shocks(trial_key: Array, n_steps: int, n_shocks: int) -> Array:
    """Generate all N(0,1) shocks for one trial.

    Each timestep receives a fresh sub-key derived deterministically
    from ``trial_key`` via ``jax.random.split``.

    Args:
        trial_key: JAX PRNGKey for this trial.
        n_steps: Number of timesteps.
        n_shocks: Total number of independent shock streams.

    Returns:
        Array of shape ``(n_steps, n_shocks)`` with standard-normal
        samples.
    """
    step_keys = jax.random.split(trial_key, n_steps)
    return jax.vmap(lambda k: jax.random.normal(k, shape=(n_shocks,)))(step_keys)


# ---------------------------------------------------------------------------
# Shock splitting
# ---------------------------------------------------------------------------


def split_shocks(raw_shocks: Array, shock_sizes: list[int]) -> list[Array]:
    """Split a combined shock array into per-model sub-arrays.

    Args:
        raw_shocks: Array of shape ``(n_steps, total_shocks)``.
        shock_sizes: Number of shocks consumed by each model,
            e.g. ``[1, 2, 1, 1]``.  Must sum to ``total_shocks``.

    Returns:
        List of arrays, each of shape ``(n_steps, n_model_shocks)``.
    """
    split_points = jnp.cumsum(jnp.array(shock_sizes[:-1]))
    return list(jnp.split(raw_shocks, split_points, axis=1))


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_trial_shocks(
    seed: int,
    n_trials: int,
    n_steps: int,
    n_shocks: int,
) -> Array:
    """Generate all shocks for every trial in a single regime.

    Convenience wrapper that creates keys and vectorises
    :func:`generate_shocks` across trials.

    Args:
        seed: Integer seed for reproducibility.
        n_trials: Number of independent trials.
        n_steps: Number of timesteps per trial.
        n_shocks: Number of shock streams per timestep.

    Returns:
        Array of shape ``(n_trials, n_steps, n_shocks)``.
    """
    # Single regime → index 0
    keys = create_rng_keys(seed, n_trials, 1)[0]
    return jax.vmap(lambda k: generate_shocks(k, n_steps, n_shocks))(keys)
