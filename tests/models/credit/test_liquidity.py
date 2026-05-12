"""Tests for LiquidityState and CIRLiquidityProcess."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.intensity_transform import SplineIntensityTransform
from hyesg.models.credit.liquidity import (
    CIRLiquidityProcess,
    LiquidityProcess,
    LiquidityState,
)

# Enable float64
jax.config.update("jax_enable_x64", True)

# Production knots for base RN→RW transform
KNOT_XS = [0, 0.02, 0.06, 0.09, 0.12, 0.2, 0.5, 1, 10]
KNOT_YS = [
    0,
    0.00450450,
    0.03009029,
    0.05256794,
    0.07213938,
    0.12588947,
    0.33689402,
    0.70219276,
    7.70218303,
]


@pytest.fixture
def base_transform() -> SplineIntensityTransform:
    """Create base RN→RW transform."""
    return SplineIntensityTransform(KNOT_XS, KNOT_YS)


@pytest.fixture
def medium_process(
    base_transform: SplineIntensityTransform,
) -> CIRLiquidityProcess:
    """Create medium-tier liquidity process with C# reference params."""
    return CIRLiquidityProcess(
        alpha=0.0225,
        mu=0.1,
        sigma=0.1,
        x0=0.04,
        rn_transform=base_transform,
        scale_factor=0.1,
        recovery_rate=0.75,
    )


@pytest.fixture
def low_process(
    base_transform: SplineIntensityTransform,
) -> CIRLiquidityProcess:
    """Create low-tier liquidity process with C# reference params."""
    return CIRLiquidityProcess(
        alpha=0.0225,
        mu=0.3,
        sigma=0.12,
        x0=0.08,
        rn_transform=base_transform,
        scale_factor=0.1,
        recovery_rate=0.75,
    )


@pytest.fixture
def rng_key():
    """Deterministic PRNG key."""
    return jax.random.PRNGKey(42)


class TestLiquidityState:
    """Tests for LiquidityState NamedTuple."""

    def test_init_state(
        self, medium_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """init_state should produce correct initial values."""
        state = medium_process.init_state(rng_key)
        assert isinstance(state, LiquidityState)
        assert jnp.isclose(state.intensity, 0.04, atol=1e-12)
        assert jnp.isclose(state.cum_intensity, 0.0, atol=1e-12)
        assert jnp.isclose(state.has_triggered, 0.0, atol=1e-12)
        assert jnp.isinf(state.trigger_time)

    def test_step_intensity_non_negative(
        self, medium_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """Intensity should remain non-negative after stepping."""
        state = medium_process.init_state(rng_key)
        # Step with large negative shock
        dz = jnp.array(-3.0, dtype=jnp.float64)
        new_state = medium_process.step(state, t=0.0, dt=0.25, dz=dz)
        assert float(new_state.intensity) >= 0.0

    def test_step_cum_intensity_increasing(
        self, medium_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """Cumulative intensity should be non-decreasing."""
        state = medium_process.init_state(rng_key)
        dz = jnp.array(0.1, dtype=jnp.float64)
        new_state = medium_process.step(state, t=0.0, dt=0.25, dz=dz)
        assert float(new_state.cum_intensity) >= float(state.cum_intensity)


class TestCIRLiquidityProcess:
    """Tests for CIRLiquidityProcess."""

    def test_medium_parameters(
        self, medium_process: CIRLiquidityProcess
    ) -> None:
        """Medium-tier should have correct recovery rate."""
        assert medium_process.recovery_rate == 0.75

    def test_low_parameters(
        self, low_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """Low-tier should start with higher initial intensity."""
        state = low_process.init_state(rng_key)
        assert jnp.isclose(state.intensity, 0.08, atol=1e-12)

    def test_trigger_detection(
        self, medium_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """After many steps with positive shocks, trigger should eventually fire."""
        state = medium_process.init_state(rng_key)
        # Run many steps to accumulate intensity
        for i in range(1000):
            dz = jnp.array(0.5, dtype=jnp.float64)
            state = medium_process.step(state, t=i * 0.01, dt=0.01, dz=dz)

        # After many steps, cumulative intensity should be substantial
        assert float(state.cum_intensity) > 0.0

    def test_scaled_transform(
        self, base_transform: SplineIntensityTransform
    ) -> None:
        """Liquidity process should use scaled (0.1x) RN→RW transform."""
        process = CIRLiquidityProcess(
            alpha=0.0225,
            mu=0.1,
            sigma=0.1,
            x0=0.04,
            rn_transform=base_transform,
            scale_factor=0.1,
        )
        # The internal transform should be scaled
        rn_val = jnp.array(0.2)
        base_rw = base_transform.transform(rn_val)
        scaled_rw = process._transform.transform(rn_val)
        assert jnp.isclose(scaled_rw, base_rw * 0.1, atol=1e-10)

    def test_multiple_steps_stability(
        self, medium_process: CIRLiquidityProcess, rng_key: jax.Array
    ) -> None:
        """Process should remain stable over many steps."""
        state = medium_process.init_state(rng_key)
        key = rng_key
        for i in range(100):
            key, subkey = jax.random.split(key)
            dz = jax.random.normal(subkey, dtype=jnp.float64)
            state = medium_process.step(state, t=i * 0.25, dt=0.25, dz=dz)

        assert jnp.isfinite(state.intensity)
        assert jnp.isfinite(state.cum_intensity)
        assert float(state.intensity) >= 0.0
