"""Tests for bond analytics — coupon schedules, YTM, duration, convexity, DV01."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.bond_analytics import (
    BondMetrics,
    _build_cashflows,
    compute_bond_metrics,
    convexity,
    dv01,
    elapsed_coupons,
    future_coupons,
    macaulay_duration,
    modified_duration,
    next_coupon,
    yield_to_maturity,
)

jax.config.update("jax_enable_x64", True)


# ─── Coupon Schedule Tests ───


class TestFutureCoupons:
    """Tests for future_coupons."""

    def test_semi_annual_from_zero(self) -> None:
        """Semi-annual coupons on a 2-year bond from t=0."""
        dates = future_coupons(maturity=2.0, coupon_freq=2, t=0.0)
        assert len(dates) == 4
        assert dates == pytest.approx([0.5, 1.0, 1.5, 2.0], abs=1e-10)

    def test_annual_from_zero(self) -> None:
        """Annual coupons on a 3-year bond from t=0."""
        dates = future_coupons(maturity=3.0, coupon_freq=1, t=0.0)
        assert len(dates) == 3
        assert dates == pytest.approx([1.0, 2.0, 3.0], abs=1e-10)

    def test_mid_life(self) -> None:
        """Coupons remaining after t=1.0 on a 3-year semi-annual bond."""
        dates = future_coupons(maturity=3.0, coupon_freq=2, t=1.0)
        assert len(dates) == 4
        assert dates[0] == pytest.approx(1.5, abs=1e-10)

    def test_at_maturity(self) -> None:
        """No future coupons at maturity."""
        dates = future_coupons(maturity=2.0, coupon_freq=2, t=2.0)
        assert len(dates) == 0

    def test_quarterly(self) -> None:
        """Quarterly coupons on a 1-year bond."""
        dates = future_coupons(maturity=1.0, coupon_freq=4, t=0.0)
        assert len(dates) == 4


class TestElapsedCoupons:
    """Tests for elapsed_coupons."""

    def test_none_elapsed_at_start(self) -> None:
        """No coupons elapsed at t=0."""
        dates = elapsed_coupons(maturity=2.0, coupon_freq=2, t=0.0)
        assert len(dates) == 0

    def test_some_elapsed(self) -> None:
        """Two coupons elapsed by t=1.0 on a semi-annual bond."""
        dates = elapsed_coupons(maturity=2.0, coupon_freq=2, t=1.0)
        assert len(dates) == 2
        assert dates == pytest.approx([0.5, 1.0], abs=1e-10)

    def test_all_elapsed_at_maturity(self) -> None:
        """All coupons elapsed at maturity."""
        dates = elapsed_coupons(maturity=2.0, coupon_freq=2, t=2.0)
        assert len(dates) == 4


class TestNextCoupon:
    """Tests for next_coupon."""

    def test_from_start(self) -> None:
        """Next coupon from t=0 is the first coupon date."""
        nc = next_coupon(maturity=2.0, coupon_freq=2, t=0.0)
        assert nc == pytest.approx(0.5, abs=1e-10)

    def test_mid_period(self) -> None:
        """Next coupon from t=0.3 on a semi-annual bond."""
        nc = next_coupon(maturity=2.0, coupon_freq=2, t=0.3)
        assert nc == pytest.approx(0.5, abs=1e-10)

    def test_at_maturity_returns_maturity(self) -> None:
        """At maturity, returns maturity itself."""
        nc = next_coupon(maturity=2.0, coupon_freq=2, t=2.0)
        assert nc == pytest.approx(2.0, abs=1e-10)


# ─── YTM Tests ───


class TestYieldToMaturity:
    """Tests for yield_to_maturity."""

    def test_par_bond_ytm_equals_coupon(self) -> None:
        """A bond priced at par has YTM ≈ coupon rate."""
        face = 100.0
        coupon = 0.05
        maturity = 5.0
        freq = 2
        # Price a par bond: build cash flows and price at coupon rate
        cfs, dates = _build_cashflows(face, coupon, maturity, freq, 0.0)
        price = sum(cf / (1.0 + coupon) ** t for cf, t in zip(cfs, dates))
        ytm = yield_to_maturity(price, face, coupon, maturity, freq)
        assert ytm == pytest.approx(coupon, abs=1e-6)

    def test_zero_coupon_bond(self) -> None:
        """ZCB YTM derived from discount price."""
        face = 100.0
        maturity = 5.0
        y_true = 0.06
        price = face / (1.0 + y_true) ** maturity
        ytm = yield_to_maturity(price, face, 0.0, maturity, freq=1)
        assert ytm == pytest.approx(y_true, abs=1e-6)

    def test_premium_bond(self) -> None:
        """Premium bond has YTM < coupon rate."""
        face = 100.0
        coupon = 0.08
        maturity = 5.0
        price = 110.0  # premium
        ytm = yield_to_maturity(price, face, coupon, maturity, freq=2)
        assert ytm < coupon

    def test_discount_bond(self) -> None:
        """Discount bond has YTM > coupon rate."""
        face = 100.0
        coupon = 0.03
        maturity = 5.0
        price = 90.0  # discount
        ytm = yield_to_maturity(price, face, coupon, maturity, freq=2)
        assert ytm > coupon


# ─── Duration Tests ───


class TestDuration:
    """Tests for Macaulay and modified duration."""

    def test_macaulay_positive(self) -> None:
        """Macaulay duration should be positive."""
        cfs = [2.5, 2.5, 2.5, 2.5, 102.5]
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        mac = macaulay_duration(0.05, cfs, times)
        assert mac > 0.0

    def test_modified_less_than_macaulay(self) -> None:
        """Modified duration < Macaulay duration for positive yield."""
        cfs = [2.5, 2.5, 2.5, 2.5, 102.5]
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        ytm = 0.05
        mac = macaulay_duration(ytm, cfs, times)
        mod = modified_duration(ytm, cfs, times)
        assert mod < mac

    def test_zero_coupon_duration_equals_maturity(self) -> None:
        """ZCB Macaulay duration = maturity."""
        cfs = [100.0]
        times = [5.0]
        mac = macaulay_duration(0.05, cfs, times)
        assert mac == pytest.approx(5.0, abs=1e-10)

    def test_duration_decreases_with_coupon(self) -> None:
        """Higher coupon → lower duration (same maturity)."""
        times = [1.0, 2.0, 3.0]
        cfs_low = [2.0, 2.0, 102.0]
        cfs_high = [8.0, 8.0, 108.0]
        dur_low = macaulay_duration(0.05, cfs_low, times)
        dur_high = macaulay_duration(0.05, cfs_high, times)
        assert dur_high < dur_low


# ─── Convexity Tests ───


class TestConvexity:
    """Tests for bond convexity."""

    def test_positive_convexity(self) -> None:
        """Convexity is positive for standard bonds."""
        cfs = [2.5, 2.5, 2.5, 2.5, 102.5]
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        conv = convexity(0.05, cfs, times)
        assert conv > 0.0

    def test_zero_coupon_convexity(self) -> None:
        """ZCB convexity = T(T+1) / (1+y)²."""
        cfs = [100.0]
        times = [5.0]
        ytm = 0.05
        conv = convexity(ytm, cfs, times)
        expected = 5.0 * 6.0 / (1.05**2)
        assert conv == pytest.approx(expected, abs=1e-6)


# ─── DV01 Tests ───


class TestDV01:
    """Tests for DV01."""

    def test_dv01_positive(self) -> None:
        """DV01 is positive for a standard bond."""
        cfs = [2.5, 2.5, 2.5, 2.5, 102.5]
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        d = dv01(0.05, cfs, times)
        assert d > 0.0

    def test_dv01_increases_with_duration(self) -> None:
        """Longer duration → higher DV01."""
        times_short = [1.0, 2.0]
        cfs_short = [5.0, 105.0]
        times_long = [1.0, 2.0, 3.0, 4.0, 5.0]
        cfs_long = [5.0, 5.0, 5.0, 5.0, 105.0]
        dv01_short = dv01(0.05, cfs_short, times_short)
        dv01_long = dv01(0.05, cfs_long, times_long)
        assert dv01_long > dv01_short


# ─── BondMetrics Container ───


class TestBondMetrics:
    """Tests for the BondMetrics container and compute_bond_metrics."""

    def test_named_tuple_fields(self) -> None:
        """BondMetrics has all expected fields."""
        bm = BondMetrics(
            price=jnp.asarray(100.0),
            ytm=jnp.asarray(0.05),
            duration=jnp.asarray(4.5),
            convexity=jnp.asarray(22.0),
            spread=jnp.asarray(0.01),
            dv01=jnp.asarray(0.045),
        )
        assert float(bm.price) == pytest.approx(100.0)
        assert float(bm.ytm) == pytest.approx(0.05)
        assert float(bm.duration) == pytest.approx(4.5)

    def test_compute_bond_metrics_par(self) -> None:
        """compute_bond_metrics produces consistent results."""
        face = 100.0
        coupon = 0.05
        maturity = 5.0
        freq = 2
        cfs, dates = _build_cashflows(face, coupon, maturity, freq, 0.0)
        price = sum(cf / (1.0 + coupon) ** t for cf, t in zip(cfs, dates))
        metrics = compute_bond_metrics(price, face, coupon, maturity, freq)
        assert float(metrics.ytm) == pytest.approx(coupon, abs=1e-5)
        assert float(metrics.duration) > 0.0
        assert float(metrics.convexity) > 0.0
        assert float(metrics.dv01) > 0.0
