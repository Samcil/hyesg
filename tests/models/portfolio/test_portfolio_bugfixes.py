"""Regression tests for portfolio bug-fixes (Phase 3, Issue #33)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hyesg.models.portfolio.analytics import PortfolioAnalytics
from hyesg.models.portfolio.portfolio import (
    _buy_and_hold_returns,
    _periodic_rebalance_returns,
)

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Bug 3: sharpe_ratio div-by-zero
# ------------------------------------------------------------------


class TestSharpeRatioDivByZero:
    def test_constant_returns_no_nan(self) -> None:
        """Constant returns → std=0 → should return 0, not NaN/inf."""
        r = jnp.array([[0.05, 0.05, 0.05, 0.05]])
        sharpe = PortfolioAnalytics.sharpe_ratio(r, risk_free=0.05)
        assert jnp.all(jnp.isfinite(sharpe))
        assert jnp.isclose(sharpe[0], 0.0, atol=1e-14)

    def test_zero_returns_zero_rf(self) -> None:
        """All-zero returns with zero risk-free → 0/0 guarded."""
        r = jnp.zeros((3, 10))
        sharpe = PortfolioAnalytics.sharpe_ratio(r, risk_free=0.0)
        assert jnp.all(jnp.isfinite(sharpe))
        assert jnp.allclose(sharpe, 0.0, atol=1e-14)

    def test_nonzero_std_unchanged(self) -> None:
        """Non-degenerate case still computes correctly."""
        r = jnp.array([[0.10, 0.05, 0.08, 0.12]])
        rf = 0.02
        excess = r - rf
        expected = jnp.mean(excess, axis=1) / jnp.std(excess, axis=1)
        sharpe = PortfolioAnalytics.sharpe_ratio(r, risk_free=rf)
        assert jnp.allclose(sharpe, expected, atol=1e-10)


# ------------------------------------------------------------------
# Bug 5: _periodic_rebalance_returns uses jax.lax.scan
# ------------------------------------------------------------------


class TestPeriodicRebalanceScan:
    def test_frequency_one_matches_constant_mix(self) -> None:
        """Rebalancing every step should reproduce constant-mix returns."""
        r_a = jnp.array([[0.10, 0.05, -0.02]])
        r_b = jnp.array([[0.02, -0.01, 0.04]])
        returns_stack = jnp.stack([r_a, r_b], axis=-1)  # (1,3,2)
        weights = jnp.array([0.6, 0.4])

        port_returns, _ = _periodic_rebalance_returns(
            returns_stack, weights, frequency=1
        )
        expected = jnp.sum(returns_stack * weights, axis=-1)
        assert jnp.allclose(port_returns, expected, atol=1e-10)

    def test_weights_reset_at_rebalance(self) -> None:
        """Weights reset to target at rebalance steps."""
        n_steps = 6
        r_a = jnp.array([[0.10] * n_steps])
        r_b = jnp.array([[0.00] * n_steps])
        returns_stack = jnp.stack([r_a, r_b], axis=-1)
        weights = jnp.array([0.5, 0.5])

        _, weights_history = _periodic_rebalance_returns(
            returns_stack, weights, frequency=3
        )
        # Step 0 and step 3 should have target weights
        assert jnp.isclose(
            weights_history[0, 0, 0], 0.5, atol=1e-10
        )
        assert jnp.isclose(
            weights_history[0, 3, 0], 0.5, atol=1e-10
        )
        # Steps 1, 2 should have drifted
        assert weights_history[0, 1, 0] > 0.5
        assert weights_history[0, 2, 0] > 0.5

    def test_output_shapes(self) -> None:
        n_trials, n_steps, n_assets = 4, 8, 3
        returns_stack = jnp.zeros((n_trials, n_steps, n_assets))
        weights = jnp.ones(n_assets) / n_assets
        port_returns, wh = _periodic_rebalance_returns(
            returns_stack, weights, frequency=2
        )
        assert port_returns.shape == (n_trials, n_steps)
        assert wh.shape == (n_trials, n_steps, n_assets)


# ------------------------------------------------------------------
# Bug 6: _buy_and_hold_returns NaN on wipeout
# ------------------------------------------------------------------


class TestBuyAndHoldWipeout:
    def test_total_wipeout_no_nan(self) -> None:
        """Complete wipeout (-100% return) should not produce NaN weights."""
        r = jnp.array([[[-1.0, -1.0]]])  # (1, 1, 2)
        weights = jnp.array([0.5, 0.5])
        _, weights_history = _buy_and_hold_returns(r, weights)
        assert jnp.all(jnp.isfinite(weights_history))

    def test_partial_wipeout_finite(self) -> None:
        """One asset wiped out, another survives."""
        r = jnp.array([[[0.10, -1.0], [0.05, 0.0]]])
        weights = jnp.array([0.5, 0.5])
        port_ret, wh = _buy_and_hold_returns(r, weights)
        assert jnp.all(jnp.isfinite(wh))
        assert jnp.all(jnp.isfinite(port_ret))

    def test_normal_case_unaffected(self) -> None:
        """Normal returns produce correct weights (no regression)."""
        r = jnp.array([[[0.10, -0.05], [0.05, 0.02]]])  # (1, 2, 2)
        weights = jnp.array([0.6, 0.4])
        _, wh = _buy_and_hold_returns(r, weights)
        # After step 0: A_val=0.6*1.1=0.66, B_val=0.4*0.95=0.38, total=1.04
        expected_w_a = 0.66 / 1.04
        assert jnp.isclose(wh[0, 0, 0], expected_w_a, atol=1e-10)
