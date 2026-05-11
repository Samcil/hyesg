"""Tests for financial pricing formulas."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.pricing import (
    black_call,
    black_implied_vol,
    black_put,
    bond_convexity,
    bond_duration,
    bond_price,
    bond_yield,
    sabr_implied_vol,
)

# Enable float64
jax.config.update("jax_enable_x64", True)


class TestBlackCall:
    """Tests for Black's model call price."""

    def test_atm_positive(self) -> None:
        """ATM call should be positive."""
        price = black_call(100.0, 100.0, 0.20, 1.0, 1.0)
        assert float(price) > 0.0

    def test_deep_itm(self) -> None:
        """Deep ITM call ≈ df * (F - K)."""
        price = black_call(200.0, 100.0, 0.20, 1.0, 1.0)
        assert float(price) == pytest.approx(100.0, rel=0.01)

    def test_zero_vol(self) -> None:
        """Zero vol: call = max(F-K, 0) * df."""
        # Use very small vol instead of zero (avoid division by zero)
        price = black_call(110.0, 100.0, 0.0001, 1.0, 1.0)
        assert float(price) == pytest.approx(10.0, rel=0.01)

    def test_increases_with_vol(self) -> None:
        """Higher vol → higher call price (ATM)."""
        p_low = black_call(100.0, 100.0, 0.10, 1.0, 1.0)
        p_high = black_call(100.0, 100.0, 0.30, 1.0, 1.0)
        assert float(p_high) > float(p_low)


class TestBlackPut:
    """Tests for Black's model put price."""

    def test_atm_positive(self) -> None:
        """ATM put should be positive."""
        price = black_put(100.0, 100.0, 0.20, 1.0, 1.0)
        assert float(price) > 0.0

    def test_deep_itm_put(self) -> None:
        """Deep ITM put ≈ df * (K - F)."""
        price = black_put(50.0, 100.0, 0.20, 1.0, 1.0)
        assert float(price) == pytest.approx(50.0, rel=0.01)


class TestPutCallParity:
    """Tests for Black put-call parity."""

    def test_parity(self) -> None:
        """C - P = df * (F - K)."""
        F, K, sigma, tau, df_val = 100.0, 95.0, 0.20, 1.0, 0.95
        call = black_call(F, K, sigma, tau, df_val)
        put = black_put(F, K, sigma, tau, df_val)
        parity = df_val * (F - K)
        assert float(call - put) == pytest.approx(parity, abs=1e-10)

    def test_parity_otm(self) -> None:
        """Put-call parity for OTM options."""
        F, K, sigma, tau, df_val = 100.0, 110.0, 0.25, 2.0, 0.90
        call = black_call(F, K, sigma, tau, df_val)
        put = black_put(F, K, sigma, tau, df_val)
        parity = df_val * (F - K)
        assert float(call - put) == pytest.approx(parity, abs=1e-10)


class TestBlackImpliedVol:
    """Tests for Black implied volatility."""

    def test_recover_vol_call(self) -> None:
        """Compute price from known vol → recover vol."""
        F, K, tau, df_val = 100.0, 100.0, 1.0, 1.0
        true_vol = 0.20
        price = black_call(F, K, true_vol, tau, df_val)
        recovered = black_implied_vol(float(price), F, K, tau, df_val, is_call=True)
        assert float(recovered) == pytest.approx(true_vol, abs=1e-6)

    def test_recover_vol_put(self) -> None:
        """Recover vol from put price."""
        F, K, tau, df_val = 100.0, 95.0, 1.0, 0.95
        true_vol = 0.25
        price = black_put(F, K, true_vol, tau, df_val)
        recovered = black_implied_vol(float(price), F, K, tau, df_val, is_call=False)
        assert float(recovered) == pytest.approx(true_vol, abs=1e-4)


class TestSabrImpliedVol:
    """Tests for SABR implied volatility."""

    def test_atm_approx_alpha(self) -> None:
        """At-the-money, vol ≈ α (for β=1, ν→0, ρ=0)."""
        alpha = 0.20
        vol = sabr_implied_vol(
            F=100.0,
            K=100.0,
            T=1.0,
            alpha=alpha,
            beta=1.0,
            rho=0.0,
            nu=0.001,  # near-zero vol-of-vol
        )
        assert float(vol) == pytest.approx(alpha, rel=0.05)

    def test_smile_shape(self) -> None:
        """Negative rho → higher vol for low strikes (skew)."""
        F = 100.0
        T = 1.0
        alpha, beta, nu = 0.20, 0.5, 0.30
        rho = -0.30

        vol_low = sabr_implied_vol(F, 80.0, T, alpha, beta, rho, nu)
        vol_atm = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        _vol_high = sabr_implied_vol(F, 120.0, T, alpha, beta, rho, nu)  # noqa: F841

        # With negative rho, low strike should have higher vol
        assert float(vol_low) > float(vol_atm)

    def test_positive_vol(self) -> None:
        """Vol should always be positive."""
        vol = sabr_implied_vol(
            F=100.0,
            K=100.0,
            T=1.0,
            alpha=0.20,
            beta=0.5,
            rho=-0.30,
            nu=0.40,
        )
        assert float(vol) > 0.0


class TestBondPrice:
    """Tests for bond pricing."""

    def test_zero_coupon_bond(self) -> None:
        """ZCB price = discount factor at maturity."""
        r = 0.05
        T = 5.0
        cashflows = jnp.array([100.0])
        times = jnp.array([T])

        def discount_fn(t: jnp.ndarray) -> jnp.ndarray:
            return jnp.exp(-r * t)

        price = bond_price(cashflows, times, discount_fn)
        expected = 100.0 * math.exp(-r * T)
        assert float(price) == pytest.approx(expected, abs=1e-10)

    def test_coupon_bond(self) -> None:
        """Coupon bond at par yield → price = par."""
        # Annual 5% coupon, 5-year maturity, priced at 5% yield
        r = 0.05
        times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        coupons = jnp.array([5.0, 5.0, 5.0, 5.0, 105.0])

        def discount_fn(t: jnp.ndarray) -> jnp.ndarray:
            return jnp.exp(-r * t)

        price = bond_price(coupons, times, discount_fn)
        # Not exactly 100 with continuous compounding, but close
        assert float(price) > 90.0
        assert float(price) < 110.0


class TestBondYield:
    """Tests for yield-to-maturity."""

    def test_zero_coupon(self) -> None:
        """ZCB yield = -ln(P/F)/T."""
        r = 0.05
        T = 5.0
        P = 100.0 * math.exp(-r * T)
        cashflows = jnp.array([100.0])
        times = jnp.array([T])

        y = bond_yield(P, cashflows, times)
        assert float(y) == pytest.approx(r, abs=1e-6)

    def test_coupon_bond_yield(self) -> None:
        """Price a bond at known yield → recover that yield."""
        true_yield = 0.06
        times = jnp.array([1.0, 2.0, 3.0])
        cashflows = jnp.array([5.0, 5.0, 105.0])
        # Price at true_yield
        discounts = jnp.exp(-true_yield * times)
        price = float(jnp.sum(cashflows * discounts))

        y = bond_yield(price, cashflows, times)
        assert float(y) == pytest.approx(true_yield, abs=1e-6)


class TestBondDuration:
    """Tests for Macaulay duration."""

    def test_zero_coupon_duration_equals_maturity(self) -> None:
        """Duration of a ZCB = maturity."""
        T = 5.0
        y = 0.05
        cashflows = jnp.array([100.0])
        times = jnp.array([T])
        P = 100.0 * math.exp(-y * T)

        dur = bond_duration(P, cashflows, times, y)
        assert float(dur) == pytest.approx(T, abs=1e-10)

    def test_coupon_bond_duration_less_than_maturity(self) -> None:
        """Coupon bond duration < maturity."""
        y = 0.05
        times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cashflows = jnp.array([5.0, 5.0, 5.0, 5.0, 105.0])
        discounts = jnp.exp(-y * times)
        P = float(jnp.sum(cashflows * discounts))

        dur = bond_duration(P, cashflows, times, y)
        assert float(dur) < 5.0
        assert float(dur) > 0.0


class TestBondConvexity:
    """Tests for bond convexity."""

    def test_zero_coupon_convexity(self) -> None:
        """ZCB convexity = maturity²."""
        T = 5.0
        y = 0.05
        cashflows = jnp.array([100.0])
        times = jnp.array([T])
        P = 100.0 * math.exp(-y * T)

        conv = bond_convexity(P, cashflows, times, y)
        assert float(conv) == pytest.approx(T**2, abs=1e-10)

    def test_positive_convexity(self) -> None:
        """Convexity should always be positive."""
        y = 0.05
        times = jnp.array([1.0, 2.0, 3.0])
        cashflows = jnp.array([5.0, 5.0, 105.0])
        discounts = jnp.exp(-y * times)
        P = float(jnp.sum(cashflows * discounts))

        conv = bond_convexity(P, cashflows, times, y)
        assert float(conv) > 0.0
