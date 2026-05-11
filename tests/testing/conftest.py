"""Shared fixtures for parity testing tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyesg.engine.output import SimulationResult
from hyesg.testing.golden_master import GoldenMaster

jax.config.update("jax_enable_x64", True)

N_TRIALS = 500
N_STEPS = 120
SEED = 42


def _make_outputs(
    seed: int,
    n_trials: int = N_TRIALS,
    n_steps: int = N_STEPS,
    *,
    mean_shift: float = 0.0,
    scale: float = 1.0,
) -> dict[str, dict[str, jax.Array]]:
    """Generate synthetic model outputs from a known seed.

    Args:
        seed: Random seed for reproducibility.
        n_trials: Number of trials.
        n_steps: Number of timesteps.
        mean_shift: Shift applied to means (to create mismatches).
        scale: Scaling factor for standard deviation.

    Returns:
        Dict mapping model names to field names to arrays.
    """
    rng = np.random.default_rng(seed)

    short_rate = rng.normal(0.03 + mean_shift, 0.01 * scale, (n_trials, n_steps))
    forward_curve = rng.normal(0.04 + mean_shift, 0.005 * scale, (n_trials, n_steps))

    equity_index = np.exp(
        np.cumsum(
            rng.normal(0.0005 + mean_shift, 0.02 * scale, (n_trials, n_steps)),
            axis=1,
        )
    )
    dividend_yield = rng.normal(
        0.02 + mean_shift, 0.003 * scale, (n_trials, n_steps)
    )

    inflation_index = np.exp(
        np.cumsum(
            rng.normal(0.0002 + mean_shift, 0.005 * scale, (n_trials, n_steps)),
            axis=1,
        )
    )

    return {
        "nominal_rates": {
            "ShortRate": jnp.array(short_rate),
            "ForwardRateCurveContinuous": jnp.array(forward_curve),
        },
        "equity": {
            "TotalReturnIndex": jnp.array(equity_index),
            "DividendYield": jnp.array(dividend_yield),
        },
        "inflation": {
            "InflationIndex": jnp.array(inflation_index),
        },
    }


@pytest.fixture()
def time_grid() -> jax.Array:
    """Deterministic monthly time grid."""
    return jnp.linspace(0.0, 10.0, N_STEPS + 1)


@pytest.fixture()
def csharp_outputs() -> dict[str, dict[str, jax.Array]]:
    """Synthetic C# reference outputs (golden master source)."""
    return _make_outputs(SEED)


@pytest.fixture()
def python_matching_outputs() -> dict[str, dict[str, jax.Array]]:
    """Python outputs drawn from the same distribution as C#."""
    return _make_outputs(SEED)


@pytest.fixture()
def python_shifted_outputs() -> dict[str, dict[str, jax.Array]]:
    """Python outputs with shifted mean — should fail parity."""
    return _make_outputs(SEED + 1, mean_shift=0.5, scale=2.0)


@pytest.fixture()
def golden_master(csharp_outputs, time_grid) -> GoldenMaster:
    """Pre-built golden master from synthetic C# data."""
    return GoldenMaster(
        name="test_csharp_v3.2",
        outputs=csharp_outputs,
        time_grid=time_grid,
        metadata={
            "csharp_version": "3.2.1",
            "created": "2024-01-15",
            "config": "test_config",
        },
    )


@pytest.fixture()
def matching_result(python_matching_outputs, time_grid) -> SimulationResult:
    """SimulationResult matching the golden master."""
    return SimulationResult(
        outputs=python_matching_outputs,
        time_grid=time_grid,
        metadata={"seed": SEED, "engine": "hyesg"},
    )


@pytest.fixture()
def shifted_result(python_shifted_outputs, time_grid) -> SimulationResult:
    """SimulationResult with shifted distributions — should fail."""
    return SimulationResult(
        outputs=python_shifted_outputs,
        time_grid=time_grid,
        metadata={"seed": SEED + 1, "engine": "hyesg"},
    )


@pytest.fixture()
def sample_result(time_grid) -> SimulationResult:
    """Small SimulationResult for basic tests."""
    rng = np.random.default_rng(99)
    outputs = {
        "model_a": {
            "field_x": jnp.array(rng.normal(0.0, 1.0, (50, 24))),
            "field_y": jnp.array(rng.normal(1.0, 0.5, (50, 24))),
        },
    }
    return SimulationResult(
        outputs=outputs,
        time_grid=jnp.linspace(0.0, 2.0, 25),
        metadata={"test": True},
    )
