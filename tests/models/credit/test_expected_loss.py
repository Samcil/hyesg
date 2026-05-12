"""Tests for P1Calculator expected loss computation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.expected_loss import P1Calculator
from hyesg.models.credit.recovery import (
    FaceValueRecovery,
    NoRecovery,
)

jax.config.update("jax_enable_x64", True)


# ─── Basic calculation ───


class TestP1Calculator:
    """Tests for P1Calculator expected loss."""

    def test_known_values(self) -> None:
        """P1 = (EAD - R*EAD) * PD with known inputs."""
        strategy = FaceValueRecovery(recovery_rate=0.4)
        calc = P1Calculator(strategy)
        exposure = jnp.array(100.0, dtype=jnp.float64)
        survival = jnp.array(0.9, dtype=jnp.float64)
        # PD = 0.1, recovery = 40, loss = (100 - 40) * 0.1 = 6.0
        result = calc.expected_loss(exposure, survival)
        assert float(result) == pytest.approx(6.0, abs=1e-10)

    def test_zero_pd_zero_loss(self) -> None:
        """Zero PD → zero expected loss."""
        strategy = FaceValueRecovery(recovery_rate=0.4)
        calc = P1Calculator(strategy)
        exposure = jnp.array(100.0, dtype=jnp.float64)
        survival = jnp.array(1.0, dtype=jnp.float64)
        result = calc.expected_loss(exposure, survival)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_full_pd_full_loss_minus_recovery(self) -> None:
        """PD=1 → loss = EAD × (1-R)."""
        strategy = FaceValueRecovery(recovery_rate=0.4)
        calc = P1Calculator(strategy)
        exposure = jnp.array(100.0, dtype=jnp.float64)
        survival = jnp.array(0.0, dtype=jnp.float64)
        # PD = 1.0, loss = (100 - 40) * 1.0 = 60.0
        result = calc.expected_loss(exposure, survival)
        assert float(result) == pytest.approx(60.0, abs=1e-10)

    def test_no_recovery_full_loss(self) -> None:
        """With NoRecovery, loss = EAD × PD."""
        strategy = NoRecovery()
        calc = P1Calculator(strategy)
        exposure = jnp.array(100.0, dtype=jnp.float64)
        survival = jnp.array(0.8, dtype=jnp.float64)
        # PD = 0.2, loss = 100 * 0.2 = 20.0
        result = calc.expected_loss(exposure, survival)
        assert float(result) == pytest.approx(20.0, abs=1e-10)

    def test_vectorised(self) -> None:
        """Works with vector inputs."""
        strategy = FaceValueRecovery(recovery_rate=0.5)
        calc = P1Calculator(strategy)
        exposure = jnp.array([100.0, 200.0, 300.0], dtype=jnp.float64)
        survival = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float64)
        result = calc.expected_loss(exposure, survival)
        # PD = [0, 0.5, 1], recovery = [50, 100, 150]
        # loss = [0, 50, 150]
        expected = jnp.array([0.0, 50.0, 150.0], dtype=jnp.float64)
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_high_recovery_low_loss(self) -> None:
        """High recovery rate should reduce expected loss."""
        strategy_low = FaceValueRecovery(recovery_rate=0.2)
        strategy_high = FaceValueRecovery(recovery_rate=0.8)
        calc_low = P1Calculator(strategy_low)
        calc_high = P1Calculator(strategy_high)
        exposure = jnp.array(100.0, dtype=jnp.float64)
        survival = jnp.array(0.5, dtype=jnp.float64)
        loss_low = float(calc_low.expected_loss(exposure, survival))
        loss_high = float(calc_high.expected_loss(exposure, survival))
        assert loss_low > loss_high
