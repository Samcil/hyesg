"""Tests for the analytic CIR++ phi integral.

Verifies that the analytic closed-form integral matches the numerical
trapezoid to high precision, and gives exact results for known cases.
"""

import jax.numpy as jnp
import pytest

from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_integral_phi,
    cir_integral_phi_analytic,
    cir_zcb_price,
    check_cir_timestep_stability,
)

# --- Test params ---
ALPHA = 0.5
MU = 0.04
SIGMA = 0.08
X0 = 0.03


def _flat_forward_curve(rate: float):
    """Create a flat forward rate curve."""

    def evaluate(t):
        return jnp.asarray(rate, dtype=jnp.float64)

    return evaluate


def _flat_zcb_curve(rate: float):
    """Create ZCB prices from a flat continuously compounded rate."""

    def evaluate(t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return jnp.exp(-rate * t)

    return evaluate


class TestAnalyticPhiIntegral:
    """Tests for cir_integral_phi_analytic."""

    def test_matches_numerical_flat_curve(self):
        """Analytic integral matches numerical trapezoid for flat curve."""
        rate = 0.05
        fwd_fn = _flat_forward_curve(rate)
        zcb_fn = _flat_zcb_curve(rate)

        t, T = 1.0, 5.0

        numerical = cir_integral_phi(t, T, ALPHA, MU, SIGMA, X0, fwd_fn)
        analytic = cir_integral_phi_analytic(t, T, ALPHA, MU, SIGMA, X0, zcb_fn)

        # Analytic is exact; numerical trapezoid has O(h²) discretisation error
        assert jnp.abs(numerical - analytic) < 1e-6, (
            f"Mismatch: numerical={numerical:.12f}, analytic={analytic:.12f}"
        )

    def test_zero_for_cir_model_curve(self):
        """When market curve IS the CIR model, phi=0 so integral=0."""

        def cir_zcb_fn(t):
            return cir_zcb_price(t, X0, ALPHA, MU, SIGMA)

        t, T = 0.5, 10.0
        result = cir_integral_phi_analytic(t, T, ALPHA, MU, SIGMA, X0, cir_zcb_fn)
        assert jnp.abs(result) < 1e-12, f"Expected ~0, got {result}"

    def test_t_equals_T(self):
        """Integral from t to t is zero."""
        zcb_fn = _flat_zcb_curve(0.05)
        result = cir_integral_phi_analytic(2.0, 2.0, ALPHA, MU, SIGMA, X0, zcb_fn)
        assert jnp.abs(result) < 1e-14

    def test_multiple_maturities(self):
        """Test across a range of maturities."""
        rate = 0.04
        fwd_fn = _flat_forward_curve(rate)
        zcb_fn = _flat_zcb_curve(rate)

        for t, T in [(0.0, 1.0), (0.0, 5.0), (1.0, 10.0), (5.0, 30.0)]:
            numerical = cir_integral_phi(t, T, ALPHA, MU, SIGMA, X0, fwd_fn)
            analytic = cir_integral_phi_analytic(t, T, ALPHA, MU, SIGMA, X0, zcb_fn)
            assert jnp.abs(numerical - analytic) < 1e-6, (
                f"t={t}, T={T}: numerical={numerical:.10f}, analytic={analytic:.10f}"
            )

    def test_additivity(self):
        """∫ₜˢ φ + ∫ₛᵀ φ = ∫ₜᵀ φ."""
        zcb_fn = _flat_zcb_curve(0.05)
        t, s, T = 1.0, 3.0, 7.0

        int_ts = cir_integral_phi_analytic(t, s, ALPHA, MU, SIGMA, X0, zcb_fn)
        int_sT = cir_integral_phi_analytic(s, T, ALPHA, MU, SIGMA, X0, zcb_fn)
        int_tT = cir_integral_phi_analytic(t, T, ALPHA, MU, SIGMA, X0, zcb_fn)

        assert jnp.abs(int_ts + int_sT - int_tT) < 1e-12


class TestTimestepStability:
    """Tests for check_cir_timestep_stability."""

    def test_stable_params(self):
        assert check_cir_timestep_stability(0.5, 0.08, 0.04, 0.25) is True

    def test_unstable_large_dt(self):
        assert check_cir_timestep_stability(0.5, 0.08, 0.04, 3.0) is False

    def test_feller_violation(self):
        # sigma^2 >> 2*alpha*mu
        assert check_cir_timestep_stability(0.1, 1.0, 0.001, 0.25) is False
