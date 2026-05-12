"""Tests for bond analytics functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.models.bond_portfolios.analytics import (  # noqa: E402, I001
    convexity,
    macaulay_duration,
    modified_duration,
    total_return,
    yield_to_maturity,
    z_spread,
)
from hyesg.models.bond_portfolios.pricing import coupon_bond_price  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────


def flat_curve(rate: float):
    """Return a flat spot-curve function."""

    def _curve(t: float) -> jax.Array:
        return jnp.asarray(rate, dtype=jnp.float64)

    return _curve


# ── Duration ─────────────────────────────────────────────────────────


class TestMacaulayDuration:
    """Macaulay duration tests."""

    def test_zcb_duration_equals_maturity(self) -> None:
        """ZCB Macaulay duration equals maturity."""
        dur = macaulay_duration(flat_curve(0.05), 0.0, 10.0, frequency=0)
        assert float(dur) == pytest.approx(10.0, abs=1e-10)

    def test_coupon_duration_less_than_maturity(self) -> None:
        """Coupon bond duration < maturity."""
        dur = macaulay_duration(flat_curve(0.05), 0.06, 10.0)
        assert float(dur) < 10.0
        assert float(dur) > 0.0

    def test_higher_coupon_lower_duration(self) -> None:
        """Higher coupon → shorter duration."""
        dur_low = macaulay_duration(flat_curve(0.05), 0.02, 10.0)
        dur_high = macaulay_duration(flat_curve(0.05), 0.08, 10.0)
        assert float(dur_high) < float(dur_low)


class TestModifiedDuration:
    """Modified duration tests."""

    def test_less_than_macaulay(self) -> None:
        """Modified duration ≤ Macaulay duration."""
        mac = macaulay_duration(flat_curve(0.05), 0.04, 10.0)
        mod = modified_duration(flat_curve(0.05), 0.04, 10.0)
        assert float(mod) <= float(mac)

    def test_positive(self) -> None:
        """Modified duration is positive."""
        mod = modified_duration(flat_curve(0.05), 0.04, 10.0)
        assert float(mod) > 0.0


# ── Convexity ────────────────────────────────────────────────────────


class TestConvexity:
    """Convexity tests."""

    def test_zcb_convexity(self) -> None:
        """ZCB convexity = T^2."""
        c = convexity(flat_curve(0.05), 0.0, 10.0, frequency=0)
        assert float(c) == pytest.approx(100.0, abs=1e-10)

    def test_positive(self) -> None:
        """Convexity is always positive."""
        c = convexity(flat_curve(0.05), 0.04, 10.0)
        assert float(c) > 0.0


# ── Yield to Maturity ────────────────────────────────────────────────


class TestYieldToMaturity:
    """YTM tests."""

    def test_zcb_ytm(self) -> None:
        """ZCB YTM = -ln(P) / T."""
        rate = 0.05
        mat = 10.0
        price = jnp.exp(jnp.asarray(-rate * mat))
        ytm = yield_to_maturity(price, 0.0, mat, frequency=0)
        assert float(ytm) == pytest.approx(rate, abs=1e-10)

    def test_coupon_bond_roundtrip(self) -> None:
        """Price → YTM → reprice roundtrips correctly."""
        rate = 0.05
        coupon = 0.04
        mat = 10.0
        curve = flat_curve(rate)
        price = coupon_bond_price(curve, coupon, mat, frequency=2)
        ytm = yield_to_maturity(price, coupon, mat, frequency=2)
        # YTM should be close to the flat curve rate
        assert float(ytm) == pytest.approx(rate, abs=1e-4)

    def test_par_bond_ytm_equals_coupon(self) -> None:
        """At par, YTM ≈ coupon rate (for continuous compounding)."""
        # Compute the price at a flat curve with rate = coupon
        rate = 0.06
        price = coupon_bond_price(flat_curve(rate), rate, 10.0, frequency=2)
        ytm = yield_to_maturity(price, rate, 10.0, frequency=2)
        assert float(ytm) == pytest.approx(rate, abs=1e-4)


# ── Z-spread ─────────────────────────────────────────────────────────


class TestZSpread:
    """Z-spread tests."""

    def test_riskfree_bond_zero_spread(self) -> None:
        """Risk-free bond priced on its own curve has zero z-spread."""
        rate = 0.05
        curve = flat_curve(rate)
        price = coupon_bond_price(curve, 0.04, 10.0, frequency=2)
        z = z_spread(price, curve, 0.04, 10.0, frequency=2)
        assert float(z) == pytest.approx(0.0, abs=1e-6)

    def test_zcb_z_spread(self) -> None:
        """ZCB z-spread for a credit-spread shifted price."""
        rate = 0.05
        spread = 0.02
        mat = 10.0
        # Price with spread
        credit_price = jnp.exp(jnp.asarray(-(rate + spread) * mat))
        z = z_spread(credit_price, flat_curve(rate), 0.0, mat, frequency=0)
        assert float(z) == pytest.approx(spread, abs=1e-10)

    def test_positive_spread_for_cheaper_bond(self) -> None:
        """Bond priced below risk-free value has positive z-spread."""
        rate = 0.05
        curve = flat_curve(rate)
        rf_price = coupon_bond_price(curve, 0.04, 10.0, frequency=2)
        cheap_price = rf_price * 0.95
        z = z_spread(cheap_price, curve, 0.04, 10.0, frequency=2)
        assert float(z) > 0.0


# ── Total Return ─────────────────────────────────────────────────────


class TestTotalReturn:
    """Total return calculation tests."""

    def test_no_change_no_income(self) -> None:
        """Zero return when price unchanged and no income."""
        ret = total_return(
            jnp.asarray(100.0),
            jnp.asarray(100.0),
            jnp.asarray(0.0),
        )
        assert float(ret) == pytest.approx(0.0, abs=1e-12)

    def test_price_appreciation(self) -> None:
        """10% price appreciation → 10% return."""
        ret = total_return(
            jnp.asarray(100.0),
            jnp.asarray(110.0),
            jnp.asarray(0.0),
        )
        assert float(ret) == pytest.approx(0.10, abs=1e-12)

    def test_income_component(self) -> None:
        """Coupon income adds to total return."""
        ret = total_return(
            jnp.asarray(100.0),
            jnp.asarray(100.0),
            jnp.asarray(5.0),
        )
        assert float(ret) == pytest.approx(0.05, abs=1e-12)

    def test_combined(self) -> None:
        """Price change + income = total return."""
        ret = total_return(
            jnp.asarray(100.0),
            jnp.asarray(105.0),
            jnp.asarray(3.0),
        )
        assert float(ret) == pytest.approx(0.08, abs=1e-12)
