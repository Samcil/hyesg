"""Tests for G2++ analytic pricing formulas."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.g2pp_analytics import (
    G2PPAnalyticParams,
    _variance_integral,
    forward_cpi,
    g2pp_zcb_price,
    il_zcb_price,
    yyiis_rate,
    zciis_rate,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def g2pp_params() -> G2PPAnalyticParams:
    """Standard G2++ test parameters."""
    return G2PPAnalyticParams(
        a1=0.05,
        a2=0.10,
        sigma1=0.01,
        sigma2=0.008,
        rho=-0.7,
    )


class TestG2PPZcbPrice:
    """Tests for G2++ zero-coupon bond pricing."""

    def test_zcb_positive(self, g2pp_params: G2PPAnalyticParams) -> None:
        """ZCB price should be positive."""
        price = g2pp_zcb_price(
            t=0.0,
            T=10.0,
            x1=jnp.float64(0.0),
            x2=jnp.float64(0.0),
            params=g2pp_params,
            phi_t=0.0,
            phi_T=-0.3,
        )
        assert float(price) > 0.0

    def test_zcb_at_maturity_is_one(self, g2pp_params: G2PPAnalyticParams) -> None:
        """P(t,t) = 1 (ZCB at its own maturity equals par)."""
        price = g2pp_zcb_price(
            t=5.0,
            T=5.0,
            x1=jnp.float64(0.01),
            x2=jnp.float64(-0.005),
            params=g2pp_params,
            phi_t=0.1,
            phi_T=0.1,
        )
        assert float(price) == pytest.approx(1.0, abs=1e-12)

    def test_zcb_decreases_with_maturity(
        self, g2pp_params: G2PPAnalyticParams
    ) -> None:
        """Longer maturity → lower ZCB price (positive rates)."""
        x1 = jnp.float64(0.02)
        x2 = jnp.float64(0.01)
        p5 = g2pp_zcb_price(0.0, 5.0, x1, x2, g2pp_params, 0.0, -0.15)
        p10 = g2pp_zcb_price(0.0, 10.0, x1, x2, g2pp_params, 0.0, -0.30)
        assert float(p5) > float(p10)

    def test_zcb_less_than_one(self, g2pp_params: G2PPAnalyticParams) -> None:
        """ZCB price < 1 for positive rates and T > t."""
        price = g2pp_zcb_price(
            t=0.0,
            T=5.0,
            x1=jnp.float64(0.02),
            x2=jnp.float64(0.01),
            params=g2pp_params,
            phi_t=0.0,
            phi_T=-0.15,
        )
        assert float(price) < 1.0

    def test_zcb_x_sensitivity(self, g2pp_params: G2PPAnalyticParams) -> None:
        """Higher x1 → lower ZCB price (positive mean reversion)."""
        p_low = g2pp_zcb_price(
            0.0, 10.0, jnp.float64(0.0), jnp.float64(0.0), g2pp_params, 0.0, -0.3
        )
        p_high = g2pp_zcb_price(
            0.0, 10.0, jnp.float64(0.05), jnp.float64(0.0), g2pp_params, 0.0, -0.3
        )
        assert float(p_low) > float(p_high)


class TestVarianceIntegral:
    """Tests for the G2++ variance integral."""

    def test_variance_positive(self, g2pp_params: G2PPAnalyticParams) -> None:
        """Variance integral should be positive for T > t."""
        V = _variance_integral(0.0, 10.0, g2pp_params)
        assert float(V) > 0.0

    def test_variance_zero_at_zero_tau(
        self, g2pp_params: G2PPAnalyticParams
    ) -> None:
        """Variance integral should be zero when T = t."""
        V = _variance_integral(5.0, 5.0, g2pp_params)
        assert float(V) == pytest.approx(0.0, abs=1e-12)

    def test_variance_increases_with_tau(
        self, g2pp_params: G2PPAnalyticParams
    ) -> None:
        """Variance integral should increase with T - t."""
        V5 = _variance_integral(0.0, 5.0, g2pp_params)
        V10 = _variance_integral(0.0, 10.0, g2pp_params)
        assert float(V10) > float(V5)


class TestForwardCPI:
    """Tests for forward CPI calculation."""

    def test_forward_cpi_consistent_with_zcbs(self) -> None:
        """Forward CPI = I(t) * P_nom / P_real."""
        inflation_index = jnp.float64(120.0)
        nominal_zcb = jnp.float64(0.90)
        real_zcb = jnp.float64(0.85)

        fwd = forward_cpi(inflation_index, nominal_zcb, real_zcb)
        expected = 120.0 * 0.90 / 0.85
        assert float(fwd) == pytest.approx(expected, abs=1e-10)

    def test_forward_cpi_positive(self) -> None:
        """Forward CPI should be positive for positive inputs."""
        fwd = forward_cpi(jnp.float64(100.0), jnp.float64(0.95), jnp.float64(0.92))
        assert float(fwd) > 0.0


class TestZCIISRate:
    """Tests for ZCIIS rate calculation."""

    def test_zciis_recovers_breakeven(self) -> None:
        """ZCIIS rate should recover breakeven inflation."""
        # If real ZCB = 0.80 and nominal ZCB = 0.75 over 10 years
        # breakeven = (0.80/0.75)^(1/10) - 1
        nominal_zcb = jnp.float64(0.75)
        real_zcb = jnp.float64(0.80)
        rate = zciis_rate(0.0, 10.0, nominal_zcb, real_zcb)
        expected = (0.80 / 0.75) ** (1.0 / 10.0) - 1.0
        assert float(rate) == pytest.approx(expected, abs=1e-10)

    def test_zciis_zero_when_equal_zcbs(self) -> None:
        """ZCIIS = 0 when real and nominal ZCBs are equal."""
        zcb = jnp.float64(0.85)
        rate = zciis_rate(0.0, 10.0, zcb, zcb)
        assert float(rate) == pytest.approx(0.0, abs=1e-12)

    def test_zciis_positive_when_real_gt_nominal(self) -> None:
        """Positive inflation when real ZCB > nominal ZCB."""
        rate = zciis_rate(0.0, 10.0, jnp.float64(0.80), jnp.float64(0.85))
        assert float(rate) > 0.0


class TestYYIISRate:
    """Tests for YYIIS rate calculation."""

    def test_yyiis_includes_convexity(self) -> None:
        """YYIIS = ZCIIS + convexity adjustment."""
        nominal_zcb = jnp.float64(0.80)
        real_zcb = jnp.float64(0.85)
        conv_adj = jnp.float64(0.001)

        base = zciis_rate(0.0, 10.0, nominal_zcb, real_zcb)
        yy = yyiis_rate(0.0, 10.0, nominal_zcb, real_zcb, conv_adj)
        assert float(yy) == pytest.approx(float(base) + 0.001, abs=1e-12)

    def test_yyiis_equals_zciis_without_convexity(self) -> None:
        """YYIIS = ZCIIS when convexity adjustment is zero."""
        nominal_zcb = jnp.float64(0.80)
        real_zcb = jnp.float64(0.85)
        conv_adj = jnp.float64(0.0)

        base = zciis_rate(0.0, 10.0, nominal_zcb, real_zcb)
        yy = yyiis_rate(0.0, 10.0, nominal_zcb, real_zcb, conv_adj)
        assert float(yy) == pytest.approx(float(base), abs=1e-12)


class TestILZcbPrice:
    """Tests for index-linked ZCB pricing."""

    def test_il_zcb_positive(self, g2pp_params: G2PPAnalyticParams) -> None:
        """IL ZCB price should be positive."""
        price = il_zcb_price(
            t=0.0,
            T=10.0,
            nominal_x1=jnp.float64(0.0),
            nominal_x2=jnp.float64(0.0),
            real_x1=jnp.float64(0.0),
            real_x2=jnp.float64(0.0),
            nominal_params=g2pp_params,
            real_params=g2pp_params,
            nominal_phi_t=0.0,
            nominal_phi_T=-0.3,
            real_phi_t=0.0,
            real_phi_T=-0.25,
            inflation_index=jnp.float64(100.0),
        )
        assert float(price) > 0.0

    def test_il_zcb_scales_with_inflation_index(
        self, g2pp_params: G2PPAnalyticParams
    ) -> None:
        """IL ZCB should scale linearly with inflation index."""
        kwargs = dict(
            t=0.0,
            T=10.0,
            nominal_x1=jnp.float64(0.0),
            nominal_x2=jnp.float64(0.0),
            real_x1=jnp.float64(0.0),
            real_x2=jnp.float64(0.0),
            nominal_params=g2pp_params,
            real_params=g2pp_params,
            nominal_phi_t=0.0,
            nominal_phi_T=-0.3,
            real_phi_t=0.0,
            real_phi_T=-0.25,
        )
        p1 = il_zcb_price(**kwargs, inflation_index=jnp.float64(100.0))
        p2 = il_zcb_price(**kwargs, inflation_index=jnp.float64(200.0))
        assert float(p2) == pytest.approx(2.0 * float(p1), rel=1e-10)
