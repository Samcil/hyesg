"""Tests for bond pricing engine."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.models.bond_portfolios.pricing import (  # noqa: E402, I001
    basket_yield,
    coupon_bond_price,
    credit_bond_price,
    index_linked_bond_price,
    zcb_price,
)


# ── Helpers ──────────────────────────────────────────────────────────


def flat_curve(rate: float):
    """Return a flat spot-curve function."""

    def _curve(t: float) -> jax.Array:
        return jnp.asarray(rate, dtype=jnp.float64)

    return _curve


def full_survival(t: float) -> jax.Array:
    """100% survival at all times (no default)."""
    return jnp.asarray(1.0, dtype=jnp.float64)


def constant_inflation(level: float = 1.0):
    """Return a constant inflation index function."""

    def _idx(t: float) -> jax.Array:
        return jnp.asarray(level, dtype=jnp.float64)

    return _idx


# ── ZCB pricing ──────────────────────────────────────────────────────


class TestZcbPrice:
    """Zero-coupon bond pricing tests."""

    def test_zero_rate(self) -> None:
        """At zero interest rate, ZCB price = 1."""
        p = zcb_price(jnp.asarray(0.0), 10.0)
        assert float(p) == pytest.approx(1.0, abs=1e-12)

    def test_positive_rate(self) -> None:
        """Standard ZCB pricing: P = exp(-r * T)."""
        r = 0.05
        t = 10.0
        p = zcb_price(jnp.asarray(r), t)
        expected = float(jnp.exp(jnp.asarray(-r * t)))
        assert float(p) == pytest.approx(expected, abs=1e-12)

    def test_zero_maturity(self) -> None:
        """At maturity, ZCB price = 1."""
        p = zcb_price(jnp.asarray(0.05), 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-12)

    def test_negative_rate(self) -> None:
        """Negative rates produce price > 1."""
        p = zcb_price(jnp.asarray(-0.01), 5.0)
        assert float(p) > 1.0


# ── Coupon bond pricing ─────────────────────────────────────────────


class TestCouponBondPrice:
    """Coupon bond pricing tests."""

    def test_par_bond_at_flat_curve(self) -> None:
        """A coupon bond with coupon = yield trades at par."""
        rate = 0.05
        curve_fn = flat_curve(rate)
        # For continuous compounding, par ≈ coupon rate
        # This is approximate for semi-annual vs continuous
        p = coupon_bond_price(curve_fn, rate, 10.0, frequency=2)
        # Should be close to 1 but not exact due to compounding mismatch
        assert 0.9 < float(p) < 1.1

    def test_zcb_fallback(self) -> None:
        """With zero coupon rate, falls back to ZCB pricing."""
        rate = 0.05
        curve_fn = flat_curve(rate)
        p = coupon_bond_price(curve_fn, 0.0, 10.0, frequency=0)
        expected = float(jnp.exp(jnp.asarray(-rate * 10.0)))
        assert float(p) == pytest.approx(expected, abs=1e-12)

    def test_higher_coupon_higher_price(self) -> None:
        """Higher coupon → higher price (same maturity and curve)."""
        curve_fn = flat_curve(0.05)
        p_low = coupon_bond_price(curve_fn, 0.02, 10.0)
        p_high = coupon_bond_price(curve_fn, 0.08, 10.0)
        assert float(p_high) > float(p_low)

    def test_longer_maturity_lower_price(self) -> None:
        """Longer maturity → lower ZCB price (positive rates)."""
        curve_fn = flat_curve(0.05)
        p_short = coupon_bond_price(curve_fn, 0.0, 5.0, frequency=0)
        p_long = coupon_bond_price(curve_fn, 0.0, 20.0, frequency=0)
        assert float(p_short) > float(p_long)


# ── Credit bond pricing ─────────────────────────────────────────────


class TestCreditBondPrice:
    """Credit-risky bond pricing tests."""

    def test_no_default_equals_riskfree(self) -> None:
        """With 100% survival and 0% recovery, credit price = risk-free."""
        curve_fn = flat_curve(0.05)
        p_rf = coupon_bond_price(curve_fn, 0.04, 10.0, frequency=2)
        p_cr = credit_bond_price(
            curve_fn, full_survival, 0.04, 10.0, 0.0, frequency=2,
        )
        assert float(p_cr) == pytest.approx(float(p_rf), abs=1e-10)

    def test_default_risk_lowers_price(self) -> None:
        """Positive default probability reduces bond price."""
        curve_fn = flat_curve(0.05)

        def risky_survival(t: float) -> jax.Array:
            return jnp.exp(jnp.asarray(-0.02 * t, dtype=jnp.float64))

        p_rf = coupon_bond_price(curve_fn, 0.04, 10.0)
        p_cr = credit_bond_price(curve_fn, risky_survival, 0.04, 10.0, 0.4)
        assert float(p_cr) < float(p_rf)

    def test_zcb_credit_bond(self) -> None:
        """ZCB credit bond pricing works."""
        curve_fn = flat_curve(0.05)
        p = credit_bond_price(curve_fn, full_survival, 0.0, 5.0, 0.4, frequency=0)
        expected = float(zcb_price(jnp.asarray(0.05), 5.0))
        assert float(p) == pytest.approx(expected, abs=1e-10)


# ── Index-linked bond pricing ───────────────────────────────────────


class TestIndexLinkedBondPrice:
    """Index-linked bond pricing tests."""

    def test_flat_inflation_no_default(self) -> None:
        """Flat inflation index + no default → same as risk-free ZCB."""
        real_curve = flat_curve(0.03)
        idx_fn = constant_inflation(1.0)
        p = index_linked_bond_price(
            real_curve, idx_fn, full_survival, 0.0, 10.0, 0.0, frequency=0,
        )
        expected = float(zcb_price(jnp.asarray(0.03), 10.0))
        assert float(p) == pytest.approx(expected, abs=1e-10)

    def test_rising_inflation_increases_price(self) -> None:
        """Rising inflation index increases IL bond price."""
        real_curve = flat_curve(0.02)

        def rising_idx(t: float) -> jax.Array:
            return jnp.asarray(1.0 + 0.03 * t, dtype=jnp.float64)

        p_flat = index_linked_bond_price(
            real_curve, constant_inflation(1.0), full_survival,
            0.0, 10.0, 0.0, frequency=0,
        )
        p_rising = index_linked_bond_price(
            real_curve, rising_idx, full_survival,
            0.0, 10.0, 0.0, frequency=0,
        )
        assert float(p_rising) > float(p_flat)


# ── Basket yield ─────────────────────────────────────────────────────


class TestBasketYield:
    """Basket yield calculation tests."""

    def test_single_bond(self) -> None:
        """Single-bond basket yield = implied ZCB yield."""
        price = jnp.exp(jnp.asarray(-0.05 * 10.0))
        y = basket_yield(
            jnp.array([price]),
            jnp.array([1.0]),
            jnp.array([10.0]),
        )
        assert float(y) == pytest.approx(0.05, abs=1e-10)

    def test_weighted_average(self) -> None:
        """Multi-bond basket yield is a weighted average."""
        prices = jnp.array([
            jnp.exp(-0.04 * 5.0),
            jnp.exp(-0.06 * 10.0),
        ])
        weights = jnp.array([0.5, 0.5])
        maturities = jnp.array([5.0, 10.0])
        y = basket_yield(prices, weights, maturities)
        expected = 0.5 * 0.04 + 0.5 * 0.06
        assert float(y) == pytest.approx(expected, abs=1e-10)
