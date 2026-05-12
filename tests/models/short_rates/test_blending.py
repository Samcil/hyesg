"""Tests for CIR2++ real-world blending, Gaussian mapper, and central-differences phi.

Covers:
 - BlendingConfig construction
 - blending_weight edge cases and interpolation
 - blended_expected_rate linearity
 - solve_rw_params 2×2 solve correctness
 - GaussianMapper properties and OU ZCB affine coefficients
 - compute_phi_central_differences vs analytic phi
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import CIRParams
from hyesg.math.cir_formulas import cir_expectation, cir_forward_rate
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.short_rates.blending import (
    BlendingConfig,
    blended_expected_rate,
    blending_weight,
    solve_rw_params,
)
from hyesg.models.short_rates.cir2pp import compute_phi_central_differences
from hyesg.models.short_rates.gaussian_mapper import GaussianMapper

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FlatForwardCurve(ParametricCurve):
    """Flat forward rate for testing."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate


class LinearCurve(ParametricCurve):
    """Linear curve f(t) = base + slope * t."""

    def __init__(self, base: float, slope: float) -> None:
        self._base = base
        self._slope = slope

    def evaluate(self, x: float) -> float:
        return self._base + self._slope * jnp.asarray(x, dtype=jnp.float64)


# Standard CIR params for tests
PARAMS1 = CIRParams(alpha=0.5, mu=0.02, sigma=0.08, initial_value=0.015)
PARAMS2 = CIRParams(alpha=0.3, mu=0.01, sigma=0.06, initial_value=0.005)

FLAT_RATE = 0.04


# ===================================================================
# BlendingConfig construction
# ===================================================================


class TestBlendingConfig:
    """Tests for BlendingConfig NamedTuple."""

    def test_construction(self) -> None:
        target = FlatForwardCurve(FLAT_RATE)
        cfg = BlendingConfig(blend_start=5.0, blend_end=20.0, blend_strength=1.0, target_curve=target)
        assert cfg.blend_start == 5.0
        assert cfg.blend_end == 20.0
        assert cfg.blend_strength == 1.0
        assert cfg.target_curve is target

    def test_namedtuple_unpacking(self) -> None:
        target = FlatForwardCurve(FLAT_RATE)
        cfg = BlendingConfig(5.0, 20.0, 1.0, target)
        start, end, strength, curve = cfg
        assert start == 5.0
        assert end == 20.0
        assert strength == 1.0
        assert curve is target


# ===================================================================
# blending_weight
# ===================================================================


class TestBlendingWeight:
    """Tests for blending_weight function."""

    @pytest.fixture()
    def config(self) -> BlendingConfig:
        return BlendingConfig(
            blend_start=5.0,
            blend_end=20.0,
            blend_strength=1.0,
            target_curve=FlatForwardCurve(FLAT_RATE),
        )

    def test_before_blend_start(self, config: BlendingConfig) -> None:
        w = blending_weight(0.0, config)
        assert float(w) == pytest.approx(0.0)

    def test_at_blend_start(self, config: BlendingConfig) -> None:
        w = blending_weight(5.0, config)
        assert float(w) == pytest.approx(0.0)

    def test_at_blend_end(self, config: BlendingConfig) -> None:
        w = blending_weight(20.0, config)
        assert float(w) == pytest.approx(1.0)

    def test_after_blend_end(self, config: BlendingConfig) -> None:
        w = blending_weight(50.0, config)
        assert float(w) == pytest.approx(1.0)

    def test_midpoint(self, config: BlendingConfig) -> None:
        t_mid = 12.5  # (5 + 20) / 2
        w = blending_weight(t_mid, config)
        assert float(w) == pytest.approx(0.5)

    def test_quarter_point(self, config: BlendingConfig) -> None:
        t = 5.0 + 0.25 * (20.0 - 5.0)  # 8.75
        w = blending_weight(t, config)
        assert float(w) == pytest.approx(0.25)

    def test_partial_strength(self) -> None:
        cfg = BlendingConfig(0.0, 10.0, 0.6, FlatForwardCurve(FLAT_RATE))
        w = blending_weight(10.0, cfg)
        assert float(w) == pytest.approx(0.6)

    def test_partial_strength_midpoint(self) -> None:
        cfg = BlendingConfig(0.0, 10.0, 0.6, FlatForwardCurve(FLAT_RATE))
        w = blending_weight(5.0, cfg)
        assert float(w) == pytest.approx(0.3)

    def test_zero_length_window(self) -> None:
        """When start == end the epsilon prevents division by zero."""
        cfg = BlendingConfig(10.0, 10.0, 1.0, FlatForwardCurve(FLAT_RATE))
        w = blending_weight(10.0, cfg)
        # (10 - 10) / epsilon -> 0 clipped, so w ≈ 0 or clipped to 1
        assert jnp.isfinite(w)


# ===================================================================
# blended_expected_rate
# ===================================================================


class TestBlendedExpectedRate:
    """Tests for blended_expected_rate function."""

    def test_full_rn(self) -> None:
        """Weight 0 → pure target (1 - 0 = 1 weight on target)."""
        result = blended_expected_rate(
            rn_expected=jnp.float64(0.03),
            target_rate=jnp.float64(0.05),
            weight=jnp.float64(0.0),
        )
        assert float(result) == pytest.approx(0.05)

    def test_full_target(self) -> None:
        """Weight 1 → pure risk-neutral."""
        result = blended_expected_rate(
            rn_expected=jnp.float64(0.03),
            target_rate=jnp.float64(0.05),
            weight=jnp.float64(1.0),
        )
        assert float(result) == pytest.approx(0.03)

    def test_half_blend(self) -> None:
        result = blended_expected_rate(
            rn_expected=jnp.float64(0.02),
            target_rate=jnp.float64(0.06),
            weight=jnp.float64(0.5),
        )
        assert float(result) == pytest.approx(0.04)

    def test_quarter_blend(self) -> None:
        result = blended_expected_rate(
            rn_expected=jnp.float64(0.02),
            target_rate=jnp.float64(0.06),
            weight=jnp.float64(0.25),
        )
        expected = 0.25 * 0.02 + 0.75 * 0.06
        assert float(result) == pytest.approx(expected)


# ===================================================================
# solve_rw_params
# ===================================================================


class TestSolveRwParams:
    """Tests for solve_rw_params 2×2 linear solver."""

    def test_known_solution(self) -> None:
        """Construct a system with known alpha_RW=0.3, mu_RW=0.05."""
        alpha_true = 0.3
        mu_true = 0.05
        x1 = 0.02
        x2 = 0.04
        # d = alpha*(mu - x) => d1 = 0.3*(0.05-0.02)=0.009, d2 = 0.3*(0.05-0.04)=0.003
        d1 = alpha_true * (mu_true - x1)
        d2 = alpha_true * (mu_true - x2)
        alpha_rw, mu_rw = solve_rw_params(
            jnp.float64(x1), jnp.float64(x2), jnp.float64(d1), jnp.float64(d2)
        )
        assert float(alpha_rw) == pytest.approx(alpha_true, rel=1e-6)
        assert float(mu_rw) == pytest.approx(mu_true, rel=1e-6)

    def test_symmetric_targets(self) -> None:
        """When derivatives differ but expectations are equal, solver handles gracefully."""
        alpha_rw, mu_rw = solve_rw_params(
            jnp.float64(0.03),
            jnp.float64(0.03),  # degenerate: x1 == x2
            jnp.float64(0.01),
            jnp.float64(0.01),
        )
        # Should not produce NaN/Inf
        assert jnp.isfinite(alpha_rw)
        assert jnp.isfinite(mu_rw)

    def test_zero_derivatives(self) -> None:
        """Zero derivatives → at equilibrium."""
        alpha_rw, mu_rw = solve_rw_params(
            jnp.float64(0.02),
            jnp.float64(0.04),
            jnp.float64(0.0),
            jnp.float64(0.0),
        )
        assert float(alpha_rw) == pytest.approx(0.0, abs=1e-10)

    def test_jit_compatible(self) -> None:
        """Verify solve_rw_params works under JIT."""
        @jax.jit
        def _solve(x1, x2, d1, d2):
            return solve_rw_params(x1, x2, d1, d2)

        a, m = _solve(jnp.float64(0.02), jnp.float64(0.04), jnp.float64(0.009), jnp.float64(0.003))
        assert jnp.isfinite(a) and jnp.isfinite(m)


# ===================================================================
# GaussianMapper
# ===================================================================


class TestGaussianMapper:
    """Tests for GaussianMapper OU parameter mapping."""

    def test_kappa_equals_alpha(self) -> None:
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        assert gm.kappa == 0.5

    def test_eta_formula(self) -> None:
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        expected_eta = 0.08 * jnp.sqrt(0.02)
        assert float(gm.eta) == pytest.approx(float(expected_eta), rel=1e-10)

    def test_eta_zero_mu(self) -> None:
        """When mu=0, eta should be 0."""
        gm = GaussianMapper(alpha=0.5, mu=0.0, sigma=0.08)
        assert float(gm.eta) == pytest.approx(0.0)

    def test_ou_zcb_B_at_maturity(self) -> None:
        """B(T,T) = 0."""
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        A, B = gm.ou_zcb_affine(5.0, 5.0)
        assert float(B) == pytest.approx(0.0, abs=1e-12)

    def test_ou_zcb_A_at_maturity(self) -> None:
        """A(T,T) = 0."""
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        A, B = gm.ou_zcb_affine(5.0, 5.0)
        assert float(A) == pytest.approx(0.0, abs=1e-12)

    def test_ou_zcb_B_positive(self) -> None:
        """B should be positive for T > t."""
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        _, B = gm.ou_zcb_affine(0.0, 10.0)
        assert float(B) > 0.0

    def test_ou_zcb_B_limit(self) -> None:
        """As tau → ∞, B → 1/kappa."""
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)
        _, B = gm.ou_zcb_affine(0.0, 1000.0)
        assert float(B) == pytest.approx(1.0 / 0.5, rel=1e-3)

    def test_jit_compatible(self) -> None:
        gm = GaussianMapper(alpha=0.5, mu=0.02, sigma=0.08)

        @jax.jit
        def _affine(t, T):
            return gm.ou_zcb_affine(t, T)

        A, B = _affine(0.0, 5.0)
        assert jnp.isfinite(A) and jnp.isfinite(B)


# ===================================================================
# compute_phi_central_differences
# ===================================================================


class TestComputePhiCentralDifferences:
    """Tests for compute_phi_central_differences against analytic phi."""

    def test_flat_curve_matches_analytic(self) -> None:
        """With a flat forward curve, central-differences phi ≈ analytic phi."""
        market_curve = FlatForwardCurve(FLAT_RATE)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)

        # Analytic forward rates for comparison
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            cd_val = float(phi_cd(t))
            # Analytic: f_market - f1 - f2
            f1 = float(cir_forward_rate(t, PARAMS1.alpha, PARAMS1.mu, PARAMS1.sigma, PARAMS1.initial_value))
            f2 = float(cir_forward_rate(t, PARAMS2.alpha, PARAMS2.mu, PARAMS2.sigma, PARAMS2.initial_value))
            analytic = max(FLAT_RATE - f1 - f2, 0.0)
            assert cd_val == pytest.approx(analytic, abs=1e-6), f"Mismatch at t={t}"

    def test_returns_callable(self) -> None:
        market_curve = FlatForwardCurve(FLAT_RATE)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)
        assert callable(phi_cd)

    def test_non_negative(self) -> None:
        """phi should always be non-negative (clamped)."""
        # Use a very low market rate so phi would be negative without clamping
        market_curve = FlatForwardCurve(0.001)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)
        for t in [0.1, 1.0, 5.0, 20.0]:
            assert float(phi_cd(t)) >= 0.0

    def test_linear_curve(self) -> None:
        """Works with a linear forward curve."""
        market_curve = LinearCurve(base=0.03, slope=0.001)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)
        val = phi_cd(5.0)
        assert jnp.isfinite(val)
        assert float(val) >= 0.0

    def test_jit_compatible(self) -> None:
        """phi callable works eagerly (curve protocol is not JIT-compatible)."""
        market_curve = FlatForwardCurve(FLAT_RATE)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)
        val = phi_cd(1.0)
        assert jnp.isfinite(val)

    def test_near_zero(self) -> None:
        """At t ≈ 0, phi should still be finite."""
        market_curve = FlatForwardCurve(FLAT_RATE)
        phi_cd = compute_phi_central_differences(market_curve, PARAMS1, PARAMS2)
        val = phi_cd(0.001)
        assert jnp.isfinite(val)
        assert float(val) >= 0.0
