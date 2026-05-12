"""Tests for recovery strategies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.recovery import (
    FaceValueRecovery,
    MarketValueRecovery,
    NoRecovery,
    RecoveryStrategy,
    TreasuryValueRecovery,
)

jax.config.update("jax_enable_x64", True)


# ─── FaceValueRecovery ───


class TestFaceValueRecovery:
    """Tests for FaceValueRecovery."""

    def test_recovery_value(self) -> None:
        """Recovery = R × face_value."""
        strategy = FaceValueRecovery(recovery_rate=0.4)
        face = jnp.array(100.0)
        market = jnp.array(80.0)
        rf = jnp.array(95.0)
        result = strategy.recovery_value(face, market, rf)
        assert float(result) == pytest.approx(40.0, abs=1e-12)

    def test_ignores_market_and_rf(self) -> None:
        """Result depends only on face value."""
        strategy = FaceValueRecovery(recovery_rate=0.5)
        face = jnp.array(200.0)
        result1 = strategy.recovery_value(face, jnp.array(50.0), jnp.array(150.0))
        result2 = strategy.recovery_value(face, jnp.array(999.0), jnp.array(1.0))
        assert float(result1) == pytest.approx(float(result2), abs=1e-12)

    def test_zero_recovery_rate(self) -> None:
        """R=0 → zero recovery."""
        strategy = FaceValueRecovery(recovery_rate=0.0)
        face = jnp.array(100.0)
        result = strategy.recovery_value(face, face, face)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_full_recovery_rate(self) -> None:
        """R=1 → full face value recovered."""
        strategy = FaceValueRecovery(recovery_rate=1.0)
        face = jnp.array(100.0)
        result = strategy.recovery_value(face, face, face)
        assert float(result) == pytest.approx(100.0, abs=1e-12)


# ─── MarketValueRecovery ───


class TestMarketValueRecovery:
    """Tests for MarketValueRecovery."""

    def test_recovery_value(self) -> None:
        """Recovery = R × market_value."""
        strategy = MarketValueRecovery(recovery_rate=0.4)
        face = jnp.array(100.0)
        market = jnp.array(80.0)
        rf = jnp.array(95.0)
        result = strategy.recovery_value(face, market, rf)
        assert float(result) == pytest.approx(32.0, abs=1e-12)

    def test_ignores_face_and_rf(self) -> None:
        """Result depends only on market value."""
        strategy = MarketValueRecovery(recovery_rate=0.6)
        market = jnp.array(50.0)
        result1 = strategy.recovery_value(jnp.array(100.0), market, jnp.array(90.0))
        result2 = strategy.recovery_value(jnp.array(200.0), market, jnp.array(10.0))
        assert float(result1) == pytest.approx(float(result2), abs=1e-12)


# ─── TreasuryValueRecovery ───


class TestTreasuryValueRecovery:
    """Tests for TreasuryValueRecovery."""

    def test_recovery_value(self) -> None:
        """Recovery = R × risk_free_value."""
        strategy = TreasuryValueRecovery(recovery_rate=0.4)
        face = jnp.array(100.0)
        market = jnp.array(80.0)
        rf = jnp.array(95.0)
        result = strategy.recovery_value(face, market, rf)
        assert float(result) == pytest.approx(38.0, abs=1e-12)

    def test_ignores_face_and_market(self) -> None:
        """Result depends only on risk-free value."""
        strategy = TreasuryValueRecovery(recovery_rate=0.3)
        rf = jnp.array(120.0)
        result1 = strategy.recovery_value(jnp.array(100.0), jnp.array(80.0), rf)
        result2 = strategy.recovery_value(jnp.array(500.0), jnp.array(10.0), rf)
        assert float(result1) == pytest.approx(float(result2), abs=1e-12)


# ─── NoRecovery ───


class TestNoRecovery:
    """Tests for NoRecovery."""

    def test_recovery_value_zero(self) -> None:
        """Always returns zero."""
        strategy = NoRecovery()
        face = jnp.array(100.0)
        market = jnp.array(80.0)
        rf = jnp.array(95.0)
        result = strategy.recovery_value(face, market, rf)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_returns_correct_shape(self) -> None:
        """Shape matches input."""
        strategy = NoRecovery()
        face = jnp.array([100.0, 200.0, 300.0])
        result = strategy.recovery_value(face, face, face)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.zeros(3))


# ─── Protocol compliance ───


class TestRecoveryProtocol:
    """Tests that all strategies implement RecoveryStrategy."""

    def test_face_value_implements_protocol(self) -> None:
        assert isinstance(FaceValueRecovery(0.4), RecoveryStrategy)

    def test_market_value_implements_protocol(self) -> None:
        assert isinstance(MarketValueRecovery(0.4), RecoveryStrategy)

    def test_treasury_value_implements_protocol(self) -> None:
        assert isinstance(TreasuryValueRecovery(0.4), RecoveryStrategy)

    def test_no_recovery_implements_protocol(self) -> None:
        assert isinstance(NoRecovery(), RecoveryStrategy)


# ─── Edge cases ───


class TestRecoveryEdgeCases:
    """Edge-case tests for recovery strategies."""

    def test_vectorised_face_value(self) -> None:
        """Works with vector inputs."""
        strategy = FaceValueRecovery(recovery_rate=0.5)
        face = jnp.array([100.0, 200.0, 300.0])
        market = jnp.array([80.0, 160.0, 240.0])
        rf = jnp.array([95.0, 190.0, 285.0])
        result = strategy.recovery_value(face, market, rf)
        expected = jnp.array([50.0, 100.0, 150.0])
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_zero_face_value(self) -> None:
        """Zero face → zero recovery."""
        strategy = FaceValueRecovery(recovery_rate=0.4)
        face = jnp.array(0.0)
        result = strategy.recovery_value(face, face, face)
        assert float(result) == pytest.approx(0.0, abs=1e-12)
