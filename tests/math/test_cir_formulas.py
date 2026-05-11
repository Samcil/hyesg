"""Tests for CIR closed-form solutions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_expectation,
    cir_forward_rate,
    cir_h,
    cir_phi_from_curves,
    cir_variance,
    cir_zcb_price,
)

# Enable float64
jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.03
SIGMA = 0.1
X0 = 0.02


class TestCirH:
    """Tests for h = √(α² + 2σ²)."""

    def test_known_value(self) -> None:
        h = cir_h(ALPHA, SIGMA)
        expected = jnp.sqrt(ALPHA**2 + 2.0 * SIGMA**2)
        assert float(h) == pytest.approx(float(expected), abs=1e-12)

    def test_zero_vol(self) -> None:
        h = cir_h(ALPHA, 0.0)
        assert float(h) == pytest.approx(ALPHA, abs=1e-12)

    def test_zero_alpha(self) -> None:
        h = cir_h(0.0, SIGMA)
        expected = jnp.sqrt(2.0) * SIGMA
        assert float(h) == pytest.approx(float(expected), abs=1e-12)


class TestCirB:
    """Tests for B(τ) coefficient."""

    def test_zero_tau(self) -> None:
        """B(0) = 0 for all parameters."""
        B = cir_B(0.0, ALPHA, SIGMA)
        assert float(B) == pytest.approx(0.0, abs=1e-12)

    def test_positive_tau(self) -> None:
        """B(τ) > 0 for τ > 0."""
        for tau in [0.1, 1.0, 5.0, 10.0]:
            B = cir_B(tau, ALPHA, SIGMA)
            assert float(B) > 0.0

    def test_vectorized(self) -> None:
        """B should work with array inputs."""
        taus = jnp.array([0.0, 1.0, 5.0, 10.0])
        Bs = cir_B(taus, ALPHA, SIGMA)
        assert Bs.shape == (4,)
        assert float(Bs[0]) == pytest.approx(0.0, abs=1e-12)
        for i in range(1, 4):
            assert float(Bs[i]) > 0.0

    def test_zero_vol_limit(self) -> None:
        """For σ→0: B(τ) = (1-e^(-ατ))/α."""
        tau = 5.0
        B = cir_B(tau, ALPHA, 0.0)
        expected = (1.0 - jnp.exp(-ALPHA * tau)) / ALPHA
        assert float(B) == pytest.approx(float(expected), abs=1e-10)


class TestCirA:
    """Tests for A(τ) coefficient."""

    def test_zero_tau(self) -> None:
        """A(0) = 1 for all parameters."""
        A = cir_A(0.0, ALPHA, MU, SIGMA)
        assert float(A) == pytest.approx(1.0, abs=1e-12)

    def test_positive_tau_less_than_one(self) -> None:
        """A(τ) in (0, 1] for positive mu and τ > 0."""
        for tau in [0.1, 1.0, 5.0]:
            A = cir_A(tau, ALPHA, MU, SIGMA)
            assert 0.0 < float(A) <= 1.0

    def test_vectorized(self) -> None:
        taus = jnp.array([0.0, 1.0, 5.0])
        As = cir_A(taus, ALPHA, MU, SIGMA)
        assert As.shape == (3,)
        assert float(As[0]) == pytest.approx(1.0, abs=1e-12)

    def test_zero_vol_limit(self) -> None:
        """For σ→0: A(τ) = exp(-μτ + μB(τ))."""
        tau = 5.0
        A = cir_A(tau, ALPHA, MU, 0.0)
        B_val = cir_B(tau, ALPHA, 0.0)
        expected = jnp.exp(-MU * tau + MU * B_val)
        assert float(A) == pytest.approx(float(expected), abs=1e-10)


class TestCirZcbPrice:
    """Tests for ZCB price P(τ, x)."""

    def test_zero_tau(self) -> None:
        """P(0, x) = 1 for all x."""
        for x in [0.01, 0.03, 0.1]:
            P = cir_zcb_price(0.0, x, ALPHA, MU, SIGMA)
            assert float(P) == pytest.approx(1.0, abs=1e-12)

    def test_positive_range(self) -> None:
        """P(τ, x) in (0, 1) for τ > 0, x > 0."""
        for tau in [1.0, 5.0, 10.0]:
            P = cir_zcb_price(tau, X0, ALPHA, MU, SIGMA)
            assert 0.0 < float(P) < 1.0

    def test_higher_x_lower_price(self) -> None:
        """Higher short rate → lower bond price."""
        P_low = cir_zcb_price(5.0, 0.01, ALPHA, MU, SIGMA)
        P_high = cir_zcb_price(5.0, 0.10, ALPHA, MU, SIGMA)
        assert float(P_high) < float(P_low)

    def test_longer_maturity_lower_price(self) -> None:
        """Longer maturity → lower bond price (positive rates)."""
        P_short = cir_zcb_price(1.0, X0, ALPHA, MU, SIGMA)
        P_long = cir_zcb_price(10.0, X0, ALPHA, MU, SIGMA)
        assert float(P_long) < float(P_short)

    def test_vectorized(self) -> None:
        taus = jnp.array([0.0, 1.0, 5.0, 10.0])
        Ps = cir_zcb_price(taus, X0, ALPHA, MU, SIGMA)
        assert Ps.shape == (4,)
        assert float(Ps[0]) == pytest.approx(1.0, abs=1e-12)

    def test_zero_vol_deterministic(self) -> None:
        """σ=0 → deterministic: P = exp(-μτ + μB(τ)) × exp(-B(τ)x)."""
        tau = 5.0
        P = cir_zcb_price(tau, X0, ALPHA, MU, 0.0)
        B_val = cir_B(tau, ALPHA, 0.0)
        A_val = cir_A(tau, ALPHA, MU, 0.0)
        expected = float(A_val * jnp.exp(-B_val * X0))
        assert float(P) == pytest.approx(expected, abs=1e-10)


class TestCirExpectation:
    """Tests for E[x(t+τ) | x(t) = x]."""

    def test_zero_tau(self) -> None:
        """E[x(0)|x] = x."""
        E = cir_expectation(0.0, X0, ALPHA, MU)
        assert float(E) == pytest.approx(X0, abs=1e-12)

    def test_long_run_mean(self) -> None:
        """As τ→∞, E → μ."""
        E = cir_expectation(1000.0, X0, ALPHA, MU)
        assert float(E) == pytest.approx(MU, abs=1e-6)

    def test_intermediate(self) -> None:
        """E[x] moves from x toward μ."""
        E = cir_expectation(1.0, X0, ALPHA, MU)
        # Should be between x and mu
        assert min(X0, MU) <= float(E) <= max(X0, MU)

    def test_vectorized(self) -> None:
        taus = jnp.array([0.0, 1.0, 10.0, 100.0])
        Es = cir_expectation(taus, X0, ALPHA, MU)
        assert Es.shape == (4,)
        assert float(Es[0]) == pytest.approx(X0, abs=1e-12)
        assert float(Es[-1]) == pytest.approx(MU, abs=1e-3)


class TestCirVariance:
    """Tests for Var[x(t+τ) | x(t) = x]."""

    def test_zero_tau(self) -> None:
        """Var(0) = 0."""
        V = cir_variance(0.0, X0, ALPHA, MU, SIGMA)
        assert float(V) == pytest.approx(0.0, abs=1e-12)

    def test_positive(self) -> None:
        """Var > 0 for τ > 0."""
        for tau in [0.1, 1.0, 5.0]:
            V = cir_variance(tau, X0, ALPHA, MU, SIGMA)
            assert float(V) > 0.0

    def test_long_run_variance(self) -> None:
        """As τ→∞, Var → μσ²/(2α)."""
        V = cir_variance(1000.0, X0, ALPHA, MU, SIGMA)
        expected = MU * SIGMA**2 / (2.0 * ALPHA)
        assert float(V) == pytest.approx(expected, abs=1e-6)


class TestCirForwardRate:
    """Tests for instantaneous forward rate."""

    def test_positive(self) -> None:
        """Forward rate should be positive for positive x."""
        for tau in [0.5, 1.0, 5.0, 10.0]:
            f = cir_forward_rate(tau, X0, ALPHA, MU, SIGMA)
            assert float(f) > 0.0

    def test_at_tau_zero(self) -> None:
        """f(0, x) should equal x (analytic identity)."""
        for x in [0.01, 0.03, 0.05, 0.10]:
            f = cir_forward_rate(0.0, x, ALPHA, MU, SIGMA)
            assert float(f) == pytest.approx(x, abs=1e-12)

    def test_long_run_limit(self) -> None:
        """Forward rate converges to a long-run value as τ→∞."""
        f_30 = float(cir_forward_rate(30.0, X0, ALPHA, MU, SIGMA))
        f_50 = float(cir_forward_rate(50.0, X0, ALPHA, MU, SIGMA))
        # Should converge — successive values close together
        assert abs(f_30 - f_50) < 0.001

    def test_agrees_with_numerical_derivative(self) -> None:
        """Analytic result should match numerical -d/dτ ln P."""
        tau = 5.0
        eps = 1e-6
        ln_p_plus = float(jnp.log(cir_zcb_price(tau + eps, X0, ALPHA, MU, SIGMA)))
        ln_p_minus = float(jnp.log(cir_zcb_price(tau - eps, X0, ALPHA, MU, SIGMA)))
        numerical = -(ln_p_plus - ln_p_minus) / (2.0 * eps)
        analytic = float(cir_forward_rate(tau, X0, ALPHA, MU, SIGMA))
        assert analytic == pytest.approx(numerical, rel=1e-6)

    def test_bond_option_not_implemented(self) -> None:
        """cir_bond_option should raise NotImplementedError."""
        from hyesg.math.cir_formulas import cir_bond_option

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            cir_bond_option(0.0, 1.0, 2.0, 0.95, X0, ALPHA, MU, SIGMA, True)


class TestCirPhiFromCurves:
    """Tests for CIR++ phi shift."""

    def test_flat_at_mu(self) -> None:
        """With flat forward at μ, phi should be approximately 0."""

        def forward_fn(t: jnp.ndarray) -> jnp.ndarray:
            return MU * jnp.ones_like(t)

        for t in [0.5, 1.0, 5.0]:
            phi = cir_phi_from_curves(t, forward_fn, ALPHA, MU, SIGMA, MU)
            # phi ≈ f_market - f_CIR; when x0=mu, CIR forward ≈ mu
            # Not exactly zero due to vol term, but should be small
            assert abs(float(phi)) < 0.05
