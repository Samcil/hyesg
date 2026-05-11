"""Shared fixtures for IO tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.output import SimulationResult

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def sample_result() -> SimulationResult:
    """A small SimulationResult for round-trip tests.

    Two models, two fields each, 4 trials × 3 steps.
    """
    n_trials, n_steps = 4, 3
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    outputs = {
        "rates": {
            "short_rate": jax.random.normal(k1, (n_trials, n_steps)),
            "forward_rate": jax.random.normal(k2, (n_trials, n_steps)),
        },
        "equity": {
            "total_return": jax.random.normal(k3, (n_trials, n_steps)),
            "dividend_yield": jax.random.normal(k4, (n_trials, n_steps)),
        },
    }
    time_grid = jnp.linspace(0.0, 1.0, n_steps + 1)
    metadata = {
        "seed": 42,
        "dt": 0.25,
        "description": "test simulation",
    }
    return SimulationResult(
        outputs=outputs,
        time_grid=time_grid,
        metadata=metadata,
    )


@pytest.fixture()
def large_result() -> SimulationResult:
    """A larger SimulationResult: 100 trials × 12 steps."""
    n_trials, n_steps = 100, 12
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)

    outputs = {
        "nominal": {
            "short_rate": jax.random.normal(k1, (n_trials, n_steps)),
            "zcb_price": jax.random.normal(k2, (n_trials, n_steps)),
        },
    }
    time_grid = jnp.linspace(0.0, 12.0, n_steps + 1)
    metadata = {"seed": 99, "n_regimes": 1}
    return SimulationResult(
        outputs=outputs,
        time_grid=time_grid,
        metadata=metadata,
    )


@pytest.fixture()
def empty_result() -> SimulationResult:
    """A SimulationResult with no outputs."""
    return SimulationResult(
        outputs={},
        time_grid=jnp.array([0.0]),
        metadata={},
    )


@pytest.fixture()
def single_field_result() -> SimulationResult:
    """A SimulationResult with a single model and single field."""
    key = jax.random.PRNGKey(7)
    outputs = {
        "cir": {
            "rate": jax.random.normal(key, (2, 5)),
        },
    }
    time_grid = jnp.linspace(0.0, 5.0, 6)
    return SimulationResult(
        outputs=outputs,
        time_grid=time_grid,
        metadata={"model": "CIR"},
    )
