"""Inner Monte Carlo utilities for nested simulation.

Provides ``MemoizingSkipForward`` for LPI swap pricing (MC within MC)
and ``double_antithetic_paths`` for variance-reduced nested simulation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class MemoizingSkipForward:
    """Skip-forward with memoised RNG state for nested MC.

    Caches key splits so that repeated forward calls from the same outer
    time re-use the same sub-keys, ensuring reproducibility in nested
    simulations.

    Args:
        outer_time: The outer simulation time at which inner MC starts.
        inner_n_trials: Number of inner Monte Carlo trials.
    """

    def __init__(self, outer_time: float, inner_n_trials: int = 1000) -> None:
        self._outer_time = outer_time
        self._inner_n_trials = inner_n_trials
        self._cache: dict[float, Any] = {}

    @property
    def outer_time(self) -> float:
        return self._outer_time

    @property
    def inner_n_trials(self) -> int:
        return self._inner_n_trials

    def forward(self, state: Any, dt: float, key: Any) -> Any:
        """Generate inner-simulation paths from the current state.

        Args:
            state: Current outer simulation state (array).
            dt: Time step for the inner simulation.
            key: JAX PRNG key.

        Returns:
            Array of shape ``(inner_n_trials,)`` representing the inner
            simulated values one step ahead.
        """
        dt_val = float(dt)
        if dt_val not in self._cache:
            self._cache[dt_val] = jax.random.split(key, self._inner_n_trials)

        sub_keys = self._cache[dt_val]
        shocks = jax.vmap(lambda k: jax.random.normal(k, shape=()))(sub_keys)
        state_arr = jnp.asarray(state)
        return state_arr + jnp.sqrt(jnp.float64(dt)) * shocks


def double_antithetic_paths(
    key: Any,
    n_outer: int,
    n_inner: int,
    n_steps: int,
) -> Any:
    """Generate outer + inner antithetic paths (4 paths per trial pair).

    For each outer trial we produce an antithetic counterpart, and for
    each inner trial we do the same, yielding 4× paths for variance
    reduction.

    Args:
        key: JAX PRNG key.
        n_outer: Number of outer trials.
        n_inner: Number of inner trials per outer trial.
        n_steps: Number of timesteps.

    Returns:
        Array of shape ``(n_outer * 2, n_inner * 2, n_steps)`` containing
        the standard + antithetic paths.
    """
    k1, k2 = jax.random.split(key)

    outer_z = jax.random.normal(k1, shape=(n_outer, n_steps))
    outer_all = jnp.concatenate([outer_z, -outer_z], axis=0)

    inner_z = jax.random.normal(k2, shape=(n_inner, n_steps))
    inner_all = jnp.concatenate([inner_z, -inner_z], axis=0)

    # Broadcast: (2*n_outer, 1, n_steps) + (1, 2*n_inner, n_steps)
    combined = outer_all[:, None, :] + inner_all[None, :, :]
    return combined
