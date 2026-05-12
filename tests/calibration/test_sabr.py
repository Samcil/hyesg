"""Tests for SABR calibration module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.calibration.sabr import (
    SabrCalibrator,
    SabrTermStructure,
    nelson_siegel_tanh,
    sabr_implied_vol_hagan,
)

jax.config.update("jax_enable_x64", True)


class TestSabrImpliedVolHagan:
    """Tests for the Hagan SABR implied vol formula."""

    def test_atm_vol_equals_alpha_lognormal(self) -> None:
        """ATM vol ≈ alpha when beta=1, nu→0, rho=0."""
        alpha = 0.20
        vol = sabr_implied_vol_hagan(
            F=100.0, K=100.0, T=1.0,
            alpha=alpha, beta=1.0, rho=0.0, nu=0.001,
        )
        assert float(vol) == pytest.approx(alpha, rel=0.05)

    def test_atm_vol_positive(self) -> None:
        """ATM vol should be positive."""
        vol = sabr_implied_vol_hagan(
            F=100.0, K=100.0, T=1.0,
            alpha=0.20, beta=0.5, rho=-0.3, nu=0.4,
        )
        assert float(vol) > 0.0

    def test_smile_shape_negative_rho(self) -> None:
        """Negative rho → higher vol for low strikes (skew)."""
        F = 100.0
        vol_low = sabr_implied_vol_hagan(F, 80.0, 1.0, 0.20, 0.5, -0.30, 0.30)
        vol_atm = sabr_implied_vol_hagan(F, F, 1.0, 0.20, 0.5, -0.30, 0.30)
        assert float(vol_low) > float(vol_atm)

    def test_smile_shape_positive_rho(self) -> None:
        """Positive rho → higher vol for high strikes."""
        F = 100.0
        vol_high = sabr_implied_vol_hagan(F, 120.0, 1.0, 0.20, 0.5, 0.30, 0.30)
        vol_atm = sabr_implied_vol_hagan(F, F, 1.0, 0.20, 0.5, 0.30, 0.30)
        assert float(vol_high) > float(vol_atm)

    def test_vol_positive_for_various_params(self) -> None:
        """Vol should be positive for reasonable parameter ranges."""
        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for rho in [-0.5, 0.0, 0.5]:
                vol = sabr_implied_vol_hagan(
                    F=100.0, K=100.0, T=1.0,
                    alpha=0.20, beta=beta, rho=rho, nu=0.30,
                )
                assert float(vol) > 0.0, f"Negative vol for beta={beta}, rho={rho}"

    def test_vol_increases_with_nu(self) -> None:
        """Higher vol-of-vol → higher ATM vol (time correction)."""
        vol_low = sabr_implied_vol_hagan(
            F=100.0, K=100.0, T=1.0,
            alpha=0.20, beta=0.5, rho=0.0, nu=0.10,
        )
        vol_high = sabr_implied_vol_hagan(
            F=100.0, K=100.0, T=1.0,
            alpha=0.20, beta=0.5, rho=0.0, nu=0.80,
        )
        assert float(vol_high) > float(vol_low)

    def test_consistency_with_existing_sabr(self) -> None:
        """Hagan formula should match the existing pricing.sabr_implied_vol."""
        from hyesg.math.pricing import sabr_implied_vol

        params = dict(F=100.0, K=95.0, T=1.0, alpha=0.20, beta=0.5, rho=-0.2, nu=0.3)
        vol_existing = sabr_implied_vol(**params)
        vol_hagan = sabr_implied_vol_hagan(**params)
        assert float(vol_hagan) == pytest.approx(float(vol_existing), rel=1e-10)

    def test_known_atm_value(self) -> None:
        """ATM with beta=0.5, check against manual calculation."""
        # For ATM (F=K=100), z=0 so z/x(z)=1
        # vol ≈ alpha / (F*K)^((1-beta)/2) * (1 + correction * T)
        # FK_beta = (F*K)^((1-beta)/2) = (100*100)^0.25 = 10000^0.25 = 10.0
        # base ≈ alpha / FK_beta = 0.20 / 10.0 = 0.02
        F = 100.0
        alpha = 0.20
        beta = 0.5
        T = 1.0
        FK_beta = (F * F) ** ((1.0 - beta) / 2.0)  # = 10.0
        base = alpha / FK_beta  # = 0.02
        vol = sabr_implied_vol_hagan(F, F, T, alpha, beta, 0.0, 0.0001)
        assert float(vol) == pytest.approx(base, rel=0.01)

    def test_short_maturity(self) -> None:
        """SABR vol should be well-behaved for short maturities."""
        vol = sabr_implied_vol_hagan(
            F=100.0, K=100.0, T=0.01,
            alpha=0.20, beta=0.5, rho=0.0, nu=0.3,
        )
        assert float(vol) > 0.0
        assert jnp.isfinite(vol)

    def test_deep_otm_positive(self) -> None:
        """Deep OTM vol should still be positive."""
        vol = sabr_implied_vol_hagan(
            F=100.0, K=50.0, T=1.0,
            alpha=0.20, beta=0.5, rho=-0.3, nu=0.4,
        )
        assert float(vol) > 0.0


class TestNelsonSiegelTanh:
    """Tests for the C# exponential-polynomial Nelson-Siegel tanh function."""

    def test_output_bounded(self) -> None:
        """Output should be in (-1, 1) for any input."""
        for b0 in [-2.0, 0.0, 2.0]:
            for b1 in [-1.0, 0.0, 1.0]:
                result = nelson_siegel_tanh(5.0, b0, b1, 0.0, 1.0)
                assert -1.0 <= float(result) <= 1.0

    def test_bounded_extreme_params(self) -> None:
        """Bounded even with extreme parameter values."""
        result = nelson_siegel_tanh(10.0, 100.0, 100.0, 100.0, 0.5)
        assert -1.0 <= float(result) <= 1.0

    def test_at_zero_time(self) -> None:
        """At t=0: tanh(beta0 + beta1) since exp(0)=1, beta2*0=0."""
        result = nelson_siegel_tanh(0.0, 0.5, 0.3, -0.1, 1.0)
        expected = jnp.tanh(0.5 + 0.3)  # beta2*0 term vanishes
        assert float(result) == pytest.approx(float(expected), abs=1e-12)

    def test_near_zero_time(self) -> None:
        """Should handle t near zero gracefully."""
        result = nelson_siegel_tanh(1e-15, 0.5, 0.3, -0.1, 1.0)
        assert jnp.isfinite(result)
        assert -1.0 <= float(result) <= 1.0

    def test_monotonic_with_beta0(self) -> None:
        """Increasing beta0 → increasing output (tanh is monotonic)."""
        r1 = nelson_siegel_tanh(5.0, 0.0, 0.0, 0.0, 1.0)
        r2 = nelson_siegel_tanh(5.0, 1.0, 0.0, 0.0, 1.0)
        assert float(r2) > float(r1)

    def test_zero_params_gives_zero(self) -> None:
        """tanh(0) = 0."""
        result = nelson_siegel_tanh(5.0, 0.0, 0.0, 0.0, 1.0)
        assert float(result) == pytest.approx(0.0, abs=1e-10)

    def test_matches_exp_poly_form(self) -> None:
        """Verify it uses beta0 + (beta1 + beta2*t)*exp(-lam*t)."""
        import math

        t, b0, b1, b2, lam = 7.0, 0.3, -0.5, 0.1, 0.2
        expected = math.tanh(b0 + (b1 + b2 * t) * math.exp(-lam * t))
        result = nelson_siegel_tanh(t, b0, b1, b2, lam)
        assert float(result) == pytest.approx(expected, abs=1e-12)


class TestSabrCalibrator:
    """Tests for the SABR calibrator."""

    def test_calibrate_atm_vol_curve(self) -> None:
        """ATM vol curve returns correct number of alphas."""
        cal = SabrCalibrator()
        vols = {1.0: 0.20, 5.0: 0.18, 10.0: 0.16}
        alphas = cal.calibrate_atm_vol_curve(vols)
        assert len(alphas) == 3

    def test_calibrate_smile_returns_structure(self) -> None:
        """Smile calibration returns a valid SabrTermStructure."""
        cal = SabrCalibrator()
        market_vols = {
            1.0: {90.0: 0.22, 100.0: 0.20, 110.0: 0.21},
            5.0: {90.0: 0.19, 100.0: 0.18, 110.0: 0.19},
        }
        result = cal.calibrate_smile(market_vols, beta=0.5)
        assert isinstance(result, SabrTermStructure)
        assert len(result.maturities) == 2
        assert result.beta == 0.5
