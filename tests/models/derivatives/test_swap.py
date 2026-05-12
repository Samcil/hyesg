"""Tests for swap, DFRN, and CDS pricing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.derivatives.cds import CDS
from hyesg.models.derivatives.dfrn import DFRN
from hyesg.models.derivatives.swap import FixedLeg, FloatingLeg, Swap

jax.config.update("jax_enable_x64", True)


class TestFixedLeg:
    """Tests for fixed leg valuation."""

    def test_fixed_leg_value(self) -> None:
        """Fixed leg PV = N * c * sum(df * delta)."""
        fixed = FixedLeg(
            notional=100.0, rate=0.05, payment_dates=(1.0, 2.0, 3.0)
        )
        swap = Swap(
            fixed_leg=fixed,
            floating_leg=FloatingLeg(100.0, "SONIA", 0.0, (1.0, 2.0, 3.0)),
        )
        r = 0.04

        def discount_curve(t: float) -> jnp.ndarray:
            return jnp.exp(-r * t)

        pv = swap.fixed_leg_value(discount_curve, t=0.0)
        # Manual: 100 * 0.05 * (exp(-0.04*1)*1 + exp(-0.04*2)*1 + exp(-0.04*3)*1)
        expected = 100.0 * 0.05 * sum(
            jnp.exp(-r * t) for t in [1.0, 2.0, 3.0]
        )
        assert float(pv) == pytest.approx(float(expected), rel=1e-6)


class TestSwap:
    """Tests for interest rate swap."""

    def test_par_swap_zero_npv(self) -> None:
        """A par swap should have approximately zero NPV."""
        r = 0.05  # flat yield curve
        dates = (1.0, 2.0, 3.0, 4.0, 5.0)

        def discount_curve(t: float) -> jnp.ndarray:
            return jnp.exp(-r * t)

        def forward_rates(t_start: float, t_end: float) -> jnp.ndarray:
            # Flat forward = continuous rate
            return jnp.float64(r)

        # Par rate for flat curve: c such that fixed leg = floating leg
        # For flat continuous: forward = r, so fixed rate = r
        fixed = FixedLeg(notional=100.0, rate=r, payment_dates=dates)
        floating = FloatingLeg(
            notional=100.0, index_ref="SONIA", spread=0.0,
            payment_dates=dates,
        )
        swap = Swap(fixed, floating)

        npv = swap.value(discount_curve, forward_rates, t=0.0)
        assert float(npv) == pytest.approx(0.0, abs=1e-6)

    def test_swap_receiver_vs_payer(self) -> None:
        """Payer and receiver swaps should have opposite signs."""
        r = 0.05
        dates = (1.0, 2.0, 3.0)

        def discount_curve(t: float) -> jnp.ndarray:
            return jnp.exp(-r * t)

        def forward_rates(t_start: float, t_end: float) -> jnp.ndarray:
            return jnp.float64(0.06)

        fixed = FixedLeg(notional=100.0, rate=0.05, payment_dates=dates)
        floating = FloatingLeg(
            notional=100.0, index_ref="SONIA", spread=0.0,
            payment_dates=dates,
        )
        swap = Swap(fixed, floating)

        npv = swap.value(discount_curve, forward_rates, t=0.0)
        # Forward (0.06) > fixed (0.05) → receive float is positive
        assert float(npv) > 0.0

    def test_swap_with_spread(self) -> None:
        """Adding spread to floating leg increases NPV."""
        r = 0.05
        dates = (1.0, 2.0, 3.0)

        def discount_curve(t: float) -> jnp.ndarray:
            return jnp.exp(-r * t)

        def forward_rates(t_start: float, t_end: float) -> jnp.ndarray:
            return jnp.float64(r)

        fixed = FixedLeg(notional=100.0, rate=r, payment_dates=dates)
        float_no_spread = FloatingLeg(100.0, "SONIA", 0.0, dates)
        float_with_spread = FloatingLeg(100.0, "SONIA", 0.01, dates)

        swap_no = Swap(fixed, float_no_spread)
        swap_with = Swap(fixed, float_with_spread)

        npv_no = swap_no.value(discount_curve, forward_rates, t=0.0)
        npv_with = swap_with.value(discount_curve, forward_rates, t=0.0)
        assert float(npv_with) > float(npv_no)


class TestDFRN:
    """Tests for Discounted Floating Rate Note."""

    def test_dfrn_value_proportional_to_notional(self) -> None:
        """DFRN value scales linearly with notional."""
        df = jnp.float64(0.95)
        d1 = DFRN(notional=100.0, spread=0.0, maturity=5.0)
        d2 = DFRN(notional=200.0, spread=0.0, maturity=5.0)
        v1 = d1.value(df, t=0.0)
        v2 = d2.value(df, t=0.0)
        assert float(v2) == pytest.approx(2.0 * float(v1), rel=1e-10)

    def test_dfrn_zero_spread_equals_notional_times_df(self) -> None:
        """With zero spread, DFRN value = notional * df."""
        df = jnp.float64(0.90)
        dfrn = DFRN(notional=100.0, spread=0.0, maturity=10.0)
        v = dfrn.value(df, t=0.0)
        assert float(v) == pytest.approx(100.0 * 0.90, abs=1e-10)

    def test_dfrn_positive_spread_increases_value(self) -> None:
        """Positive spread → higher value than zero spread."""
        df = jnp.float64(0.90)
        d_no_spread = DFRN(notional=100.0, spread=0.0, maturity=10.0)
        d_with_spread = DFRN(notional=100.0, spread=0.01, maturity=10.0)
        v_no = d_no_spread.value(df, t=0.0)
        v_with = d_with_spread.value(df, t=0.0)
        assert float(v_with) > float(v_no)


class TestCDS:
    """Tests for Credit Default Swap."""

    def test_cds_fair_spread_zero_value(self) -> None:
        """At fair spread, CDS value should be approximately zero."""
        n_periods = 10
        survival_prob = jnp.array([0.99**i for i in range(1, n_periods + 1)])
        discount_factors = jnp.array(
            [jnp.exp(-0.03 * i) for i in range(1, n_periods + 1)]
        )

        # Compute fair spread
        dt = 10.0 / n_periods
        premium_annuity = jnp.sum(survival_prob * discount_factors * dt)
        surv_shifted = jnp.concatenate(
            [jnp.array([1.0]), survival_prob[:-1]]
        )
        default_prob = surv_shifted - survival_prob
        protection_value = 0.6 * jnp.sum(default_prob * discount_factors)
        fair_spread = float(protection_value / premium_annuity)

        cds = CDS("Corp_A", spread=fair_spread, notional=100.0, maturity=10.0)
        mtm = cds.value(survival_prob, discount_factors, t=0.0)
        assert float(mtm) == pytest.approx(0.0, abs=0.5)

    def test_cds_value_from_survival_probs(self) -> None:
        """CDS value computed from explicit survival probabilities."""
        cds = CDS("Corp_B", spread=0.01, notional=100.0, maturity=5.0)
        n_periods = 5
        survival_prob = jnp.array([0.98, 0.96, 0.94, 0.92, 0.90])
        discount_factors = jnp.array(
            [jnp.exp(-0.03 * i) for i in range(1, n_periods + 1)]
        )
        mtm = cds.value(survival_prob, discount_factors, t=0.0)
        # Should be finite
        assert jnp.isfinite(mtm)

    def test_cds_higher_spread_lower_value(self) -> None:
        """Higher premium spread → lower value for protection buyer."""
        n_periods = 5
        survival_prob = jnp.array([0.98, 0.96, 0.94, 0.92, 0.90])
        discount_factors = jnp.array(
            [jnp.exp(-0.03 * i) for i in range(1, n_periods + 1)]
        )

        cds_low = CDS("Corp", spread=0.005, notional=100.0, maturity=5.0)
        cds_high = CDS("Corp", spread=0.05, notional=100.0, maturity=5.0)
        v_low = cds_low.value(survival_prob, discount_factors, t=0.0)
        v_high = cds_high.value(survival_prob, discount_factors, t=0.0)
        # Higher spread means buyer pays more → lower MTM
        assert float(v_high) < float(v_low)

    def test_cds_notional_scaling(self) -> None:
        """CDS value should scale linearly with notional."""
        n_periods = 5
        survival_prob = jnp.array([0.98, 0.96, 0.94, 0.92, 0.90])
        discount_factors = jnp.array(
            [jnp.exp(-0.03 * i) for i in range(1, n_periods + 1)]
        )

        cds1 = CDS("Corp", spread=0.01, notional=100.0, maturity=5.0)
        cds2 = CDS("Corp", spread=0.01, notional=200.0, maturity=5.0)
        v1 = cds1.value(survival_prob, discount_factors, t=0.0)
        v2 = cds2.value(survival_prob, discount_factors, t=0.0)
        assert float(v2) == pytest.approx(2.0 * float(v1), rel=1e-10)
