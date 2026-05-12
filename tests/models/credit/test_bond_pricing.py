"""Tests for CouponBondPricer and IndexLinkedBondPricer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.bond_pricing import CouponBondPricer, IndexLinkedBondPricer

# Enable float64
jax.config.update("jax_enable_x64", True)


def _flat_zcb(rate: float = 0.05):
    """Create a flat ZCB pricing function: P(T) = exp(-r*T)."""

    def zcb_fn(t: float) -> jax.Array:
        return jnp.exp(-rate * t)

    return zcb_fn


def _flat_survival(hazard: float = 0.0):
    """Create a flat survival function: S(T) = exp(-h*T)."""

    def surv_fn(t: float) -> jax.Array:
        return jnp.exp(-hazard * t)

    return surv_fn


def _flat_inflation(growth: float = 0.02):
    """Create a flat inflation index: I(T) = exp(g*T)."""

    def inflation_fn(t: float) -> jax.Array:
        return jnp.exp(growth * t)

    return inflation_fn


class TestCouponBondPricer:
    """Tests for CouponBondPricer."""

    def test_zero_coupon_matches_zcb(self) -> None:
        """A zero-coupon bond with no default risk should match ZCB."""
        pricer = CouponBondPricer(frequency=2)
        zcb_fn = _flat_zcb(rate=0.05)
        surv_fn = _flat_survival(hazard=0.0)  # no default risk

        price = pricer.price(
            coupon_rate=0.0,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        expected_zcb = jnp.exp(-0.05 * 5.0)
        assert jnp.isclose(price, expected_zcb, atol=1e-6)

    def test_coupon_increases_price(self) -> None:
        """Adding coupons should increase the bond price."""
        pricer = CouponBondPricer(frequency=2)
        zcb_fn = _flat_zcb(rate=0.05)
        surv_fn = _flat_survival(hazard=0.0)

        price_zero = pricer.price(
            coupon_rate=0.0,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        price_coupon = pricer.price(
            coupon_rate=0.05,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        assert float(price_coupon) > float(price_zero)

    def test_default_reduces_price(self) -> None:
        """Credit risk (positive hazard rate) should reduce the price."""
        pricer = CouponBondPricer(frequency=2)
        zcb_fn = _flat_zcb(rate=0.05)

        price_safe = pricer.price(
            coupon_rate=0.05,
            maturity=5.0,
            survival_fn=_flat_survival(hazard=0.0),
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        price_risky = pricer.price(
            coupon_rate=0.05,
            maturity=5.0,
            survival_fn=_flat_survival(hazard=0.05),
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        assert float(price_risky) < float(price_safe)

    def test_recovery_increases_risky_price(self) -> None:
        """Higher recovery rate should increase a risky bond's price."""
        pricer = CouponBondPricer(frequency=2)
        zcb_fn = _flat_zcb(rate=0.05)
        surv_fn = _flat_survival(hazard=0.05)

        price_low_recovery = pricer.price(
            coupon_rate=0.05,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.0,
            t=0.0,
        )

        price_high_recovery = pricer.price(
            coupon_rate=0.05,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.5,
            t=0.0,
        )

        assert float(price_high_recovery) > float(price_low_recovery)

    def test_expired_bond(self) -> None:
        """An expired bond (t > maturity) should return 1.0 (face value)."""
        pricer = CouponBondPricer(frequency=2)
        price = pricer.price(
            coupon_rate=0.05,
            maturity=3.0,
            survival_fn=_flat_survival(),
            zcb_fn=_flat_zcb(),
            recovery_rate=0.35,
            t=5.0,
        )
        assert jnp.isclose(price, 1.0, atol=1e-8)

    def test_price_positive(self) -> None:
        """Bond price should always be positive."""
        pricer = CouponBondPricer(frequency=2)
        price = pricer.price(
            coupon_rate=0.05,
            maturity=10.0,
            survival_fn=_flat_survival(hazard=0.1),
            zcb_fn=_flat_zcb(rate=0.05),
            recovery_rate=0.35,
            t=0.0,
        )
        assert float(price) > 0.0


class TestIndexLinkedBondPricer:
    """Tests for IndexLinkedBondPricer."""

    def test_combines_credit_and_inflation(self) -> None:
        """Index-linked price should exceed nominal price when inflation > 0."""
        coupon_pricer = CouponBondPricer(frequency=2)
        il_pricer = IndexLinkedBondPricer(frequency=2)

        zcb_fn = _flat_zcb(rate=0.05)
        surv_fn = _flat_survival(hazard=0.0)
        inflation_fn = _flat_inflation(growth=0.02)

        nominal_price = coupon_pricer.price(
            coupon_rate=0.03,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        il_price = il_pricer.price(
            coupon_rate=0.03,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            inflation_index_fn=inflation_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        # With positive inflation, index-linked should be worth more
        assert float(il_price) > float(nominal_price)

    def test_no_inflation_matches_nominal(self) -> None:
        """With zero inflation, index-linked price should match nominal."""
        coupon_pricer = CouponBondPricer(frequency=2)
        il_pricer = IndexLinkedBondPricer(frequency=2)

        zcb_fn = _flat_zcb(rate=0.05)
        surv_fn = _flat_survival(hazard=0.0)
        flat_index = _flat_inflation(growth=0.0)  # no inflation

        nominal_price = coupon_pricer.price(
            coupon_rate=0.03,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            recovery_rate=0.35,
            t=0.0,
        )

        il_price = il_pricer.price(
            coupon_rate=0.03,
            maturity=5.0,
            survival_fn=surv_fn,
            zcb_fn=zcb_fn,
            inflation_index_fn=flat_index,
            recovery_rate=0.35,
            t=0.0,
        )

        assert jnp.isclose(il_price, nominal_price, atol=1e-6)

    def test_expired_bond(self) -> None:
        """An expired index-linked bond should return 1.0."""
        il_pricer = IndexLinkedBondPricer(frequency=2)
        price = il_pricer.price(
            coupon_rate=0.03,
            maturity=3.0,
            survival_fn=_flat_survival(),
            zcb_fn=_flat_zcb(),
            inflation_index_fn=_flat_inflation(),
            recovery_rate=0.35,
            t=5.0,
        )
        assert jnp.isclose(price, 1.0, atol=1e-8)
