"""Tests for SABR parametric curves.

Reference values validated against C# ``Hymans.FinancialMaths``
``SabrWorkAroundTests.cs`` using the same parameters:
    beta0=-0.424189309479046, beta1=-0.810700014626725,
    beta2=0.15251116926378, lam=0.1
"""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves.sabr import (
    SabrAtmVolCurve,
    SabrNuCurve,
    SabrRhoCurve,
    _nelson_siegel_exp_poly,
)

# C# test parameters from SabrWorkAroundTests.cs
_B0 = -0.424189309479046
_B1 = -0.810700014626725
_B2 = 0.15251116926378
_LAM = 0.1
_MAX_NU = math.sqrt(24.0 / 50.0) * 0.99


class TestNelsonSiegelExpPoly:
    """Tests for the exponential-polynomial NS base function."""

    def test_at_zero(self) -> None:
        """At x=0, exp(-lam*0)=1, so NS(0) = beta0 + beta1."""
        val = _nelson_siegel_exp_poly(0.0, _B0, _B1, _B2, _LAM)
        assert val == pytest.approx(_B0 + _B1, abs=1e-14)

    def test_at_large_x(self) -> None:
        """As x→∞, exp term → 0, so NS → beta0."""
        val = _nelson_siegel_exp_poly(1000.0, _B0, _B1, _B2, _LAM)
        assert val == pytest.approx(_B0, abs=1e-10)

    def test_at_10(self) -> None:
        """Manual calculation at x=10."""
        x = 10.0
        expected = _B0 + (_B1 + _B2 * x) * math.exp(-_LAM * x)
        val = _nelson_siegel_exp_poly(x, _B0, _B1, _B2, _LAM)
        assert val == pytest.approx(expected, abs=1e-14)


class TestSabrAtmVolCurve:
    """Tests for SabrAtmVolCurve matching C# SabrAtmVolCurve."""

    def test_at_zero_maturity(self) -> None:
        """At T=0: (tanh(beta0 + beta1) + 1) / 2."""
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        ns0 = _B0 + _B1
        expected = (math.tanh(ns0) + 1.0) / 2.0
        assert c.evaluate(0.0) == pytest.approx(expected, abs=1e-14)

    def test_output_in_zero_one(self) -> None:
        """Output must be in (0, 1) for all maturities."""
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        for t in [0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
            val = c.evaluate(t)
            assert 0.0 < val < 1.0

    def test_matches_csharp_formula_at_m10(self) -> None:
        """Match C# reference: (tanh(NS(10)) + 1) / 2."""
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        ns10 = _nelson_siegel_exp_poly(10.0, _B0, _B1, _B2, _LAM)
        expected = (math.tanh(ns10) + 1.0) / 2.0
        assert c.evaluate(10.0) == pytest.approx(expected, abs=1e-14)

    def test_negative_maturity_treated_as_zero(self) -> None:
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        assert c.evaluate(-5.0) == c.evaluate(0.0)

    def test_parameters_property(self) -> None:
        c = SabrAtmVolCurve(0.1, 0.2, 0.3, 0.4)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4)

    def test_callable_syntax(self) -> None:
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        assert c(5.0) == c.evaluate(5.0)

    def test_parity_with_csharp_loop(self) -> None:
        """Validate all maturities 1..100 match C# formula."""
        c = SabrAtmVolCurve(_B0, _B1, _B2, _LAM)
        for m in range(1, 101):
            ns = _nelson_siegel_exp_poly(float(m), _B0, _B1, _B2, _LAM)
            expected = (math.tanh(ns) + 1.0) / 2.0
            assert c.evaluate(float(m)) == pytest.approx(
                expected, abs=1e-14
            ), f"Mismatch at m={m}"


class TestSabrNuCurve:
    """Tests for SabrNuCurve matching C# SabrNuCurve."""

    def test_at_zero_maturity(self) -> None:
        """At T=0: max_nu * (tanh(beta0 + beta1) + 1) / 2."""
        c = SabrNuCurve(_B0, _B1, _B2, _LAM, _MAX_NU)
        ns0 = _B0 + _B1
        expected = _MAX_NU * (math.tanh(ns0) + 1.0) / 2.0
        assert c.evaluate(0.0) == pytest.approx(expected, abs=1e-14)

    def test_output_in_zero_maxnu(self) -> None:
        """Output must be in (0, max_nu) for all maturities."""
        c = SabrNuCurve(_B0, _B1, _B2, _LAM, _MAX_NU)
        for t in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            val = c.evaluate(t)
            assert 0.0 < val < _MAX_NU

    def test_matches_csharp_formula_at_m10(self) -> None:
        """Match C# reference: maxNu * (tanh(NS(10)) + 1) / 2."""
        c = SabrNuCurve(_B0, _B1, _B2, _LAM, _MAX_NU)
        ns10 = _nelson_siegel_exp_poly(10.0, _B0, _B1, _B2, _LAM)
        expected = _MAX_NU * (math.tanh(ns10) + 1.0) / 2.0
        assert c.evaluate(10.0) == pytest.approx(expected, abs=1e-14)

    def test_parameters_property(self) -> None:
        c = SabrNuCurve(0.1, 0.2, 0.3, 0.4, 0.5)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4, 0.5)

    def test_parity_with_csharp_loop(self) -> None:
        """Validate all maturities 1..100 match C# formula."""
        c = SabrNuCurve(_B0, _B1, _B2, _LAM, _MAX_NU)
        for m in range(1, 101):
            ns = _nelson_siegel_exp_poly(float(m), _B0, _B1, _B2, _LAM)
            expected = _MAX_NU * (math.tanh(ns) + 1.0) / 2.0
            assert c.evaluate(float(m)) == pytest.approx(
                expected, abs=1e-14
            ), f"Mismatch at m={m}"


class TestSabrRhoCurve:
    """Tests for SabrRhoCurve matching C# SabrRhoCurve."""

    def test_at_zero_maturity(self) -> None:
        """At T=0: tanh(beta0 + beta1) * 0.95."""
        c = SabrRhoCurve(_B0, _B1, _B2, _LAM)
        ns0 = _B0 + _B1
        expected = math.tanh(ns0) * 0.95
        assert c.evaluate(0.0) == pytest.approx(expected, abs=1e-14)

    def test_output_bounded_by_rho_cap(self) -> None:
        """Output must be in (-0.95, 0.95) with default cap."""
        c = SabrRhoCurve(_B0, _B1, _B2, _LAM)
        for t in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            val = c.evaluate(t)
            assert -0.95 < val < 0.95

    def test_custom_rho_cap(self) -> None:
        """Custom rho_cap bounds the output differently."""
        c = SabrRhoCurve(_B0, _B1, _B2, _LAM, rho_cap=0.80)
        for t in [0.0, 5.0, 50.0]:
            val = c.evaluate(t)
            assert -0.80 < val < 0.80

    def test_matches_csharp_formula_at_m10(self) -> None:
        """Match C# reference: tanh(NS(10)) * 0.95."""
        c = SabrRhoCurve(_B0, _B1, _B2, _LAM)
        ns10 = _nelson_siegel_exp_poly(10.0, _B0, _B1, _B2, _LAM)
        expected = math.tanh(ns10) * 0.95
        assert c.evaluate(10.0) == pytest.approx(expected, abs=1e-14)

    def test_parameters_property(self) -> None:
        c = SabrRhoCurve(0.1, 0.2, 0.3, 0.4)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4, 0.95)

    def test_parity_with_csharp_loop(self) -> None:
        """Validate all maturities 1..100 match C# formula."""
        c = SabrRhoCurve(_B0, _B1, _B2, _LAM)
        for m in range(1, 101):
            ns = _nelson_siegel_exp_poly(float(m), _B0, _B1, _B2, _LAM)
            expected = math.tanh(ns) * 0.95
            assert c.evaluate(float(m)) == pytest.approx(
                expected, abs=1e-14
            ), f"Mismatch at m={m}"

    def test_imports_from_package(self) -> None:
        from hyesg.math.curves import SabrAtmVolCurve as Atm
        from hyesg.math.curves import SabrNuCurve as Nu
        from hyesg.math.curves import SabrRhoCurve as Rho

        assert Atm is not None
        assert Nu is not None
        assert Rho is not None
