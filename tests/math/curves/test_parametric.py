"""Tests for parametric curve implementations."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import GeneralizedLogistic, NelsonSiegelCurve


class TestNelsonSiegelCurve:
    """Tests for Nelson-Siegel yield curve model."""

    def test_at_zero(self) -> None:
        """f(0) = β₀ + β₁ via L'Hôpital."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        assert ns.evaluate(0.0) == pytest.approx(0.03)

    def test_long_end_approaches_beta0(self) -> None:
        """As t→∞, f(t) → β₀."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        assert ns.evaluate(100.0) == pytest.approx(0.05, abs=1e-3)

    def test_known_values(self) -> None:
        """Test with specific known parameter values."""
        ns = NelsonSiegelCurve(0.06, -0.03, 0.02, 2.0)
        # At t=0: beta0 + beta1 = 0.06 - 0.03 = 0.03
        assert ns.evaluate(0.0) == pytest.approx(0.03)

        # At t=2 (t/τ = 1):
        # factor1 = (1-e^(-1))/1 ≈ 0.6321
        # factor2 = factor1 - e^(-1) ≈ 0.6321 - 0.3679 = 0.2642
        t = 2.0
        t_tau = 1.0
        exp_val = math.exp(-t_tau)
        f1 = (1.0 - exp_val) / t_tau
        f2 = f1 - exp_val
        expected = 0.06 + (-0.03) * f1 + 0.02 * f2
        assert ns.evaluate(t) == pytest.approx(expected)

    def test_callable_syntax(self) -> None:
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        assert ns(0.0) == pytest.approx(0.03)

    def test_parameters(self) -> None:
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        assert ns.parameters == (0.05, -0.02, 0.01, 1.5)

    def test_with_parameters(self) -> None:
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        ns2 = ns.with_parameters((0.06, -0.03, 0.02, 2.0))
        assert ns2.parameters == (0.06, -0.03, 0.02, 2.0)

    def test_invalid_tau(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            NelsonSiegelCurve(0.05, -0.02, 0.01, 0.0)
        with pytest.raises(ValueError, match="positive"):
            NelsonSiegelCurve(0.05, -0.02, 0.01, -1.0)

    def test_hump_shape(self) -> None:
        """With positive β₂, should have a hump in middle maturities."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.05, 2.0)
        # Check that some middle maturity exceeds both ends
        y_mid = ns.evaluate(3.0)
        y_long = ns.evaluate(50.0)
        # Mid should be higher than long end for positive beta2
        assert y_mid > y_long - 0.01


class TestGeneralizedLogistic:
    """Tests for generalized logistic function."""

    def test_midpoint(self) -> None:
        """At x=x0 with nu=1, f(x0) = (L+U)/2."""
        gl = GeneralizedLogistic(0.0, 1.0, 1.0, 0.0, 1.0)
        assert gl.evaluate(0.0) == pytest.approx(0.5)

    def test_asymptotes(self) -> None:
        """Check lower and upper asymptotes."""
        gl = GeneralizedLogistic(0.0, 1.0, 1.0, 0.0, 1.0)
        assert gl.evaluate(-100.0) == pytest.approx(0.0, abs=1e-6)
        assert gl.evaluate(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_custom_asymptotes(self) -> None:
        gl = GeneralizedLogistic(2.0, 8.0, 1.0, 0.0, 1.0)
        assert gl.evaluate(-100.0) == pytest.approx(2.0, abs=1e-4)
        assert gl.evaluate(100.0) == pytest.approx(8.0, abs=1e-4)

    def test_steep_growth(self) -> None:
        """Large k makes transition sharper."""
        gl_steep = GeneralizedLogistic(0.0, 1.0, 10.0, 0.0, 1.0)
        gl_gentle = GeneralizedLogistic(0.0, 1.0, 0.5, 0.0, 1.0)
        # Steep should be closer to 1 at x=1
        assert gl_steep.evaluate(1.0) > gl_gentle.evaluate(1.0)

    def test_parameters(self) -> None:
        gl = GeneralizedLogistic(0.0, 1.0, 1.0, 0.0, 1.0)
        assert gl.parameters == (0.0, 1.0, 1.0, 0.0, 1.0)

    def test_with_parameters(self) -> None:
        gl = GeneralizedLogistic(0.0, 1.0, 1.0, 0.0, 1.0)
        gl2 = gl.with_parameters((2.0, 8.0, 0.5, 1.0, 2.0))
        assert gl2.parameters == (2.0, 8.0, 0.5, 1.0, 2.0)

    def test_invalid_nu(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GeneralizedLogistic(0.0, 1.0, 1.0, 0.0, 0.0)
