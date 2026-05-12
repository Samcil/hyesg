"""Tests for SABR parametric curves."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves.sabr import SabrAtmVolCurve, SabrNuCurve, SabrRhoCurve


class TestSabrAtmVolCurve:
    """Tests for SabrAtmVolCurve."""

    def test_at_zero_maturity(self) -> None:
        c = SabrAtmVolCurve(0.5, 0.3, 0.0, 1.0)
        # raw(0) = beta0 + beta1 = 0.8
        expected = math.tanh(0.8)
        assert c.evaluate(0.0) == pytest.approx(expected)

    def test_at_zero_with_alpha_beta(self) -> None:
        c = SabrAtmVolCurve(0.5, 0.3, 0.0, 1.0, alpha=0.1, beta_scale=2.0)
        expected = 0.1 + 2.0 * math.tanh(0.8)
        assert c.evaluate(0.0) == pytest.approx(expected)

    def test_at_large_maturity_converges_to_beta0(self) -> None:
        c = SabrAtmVolCurve(0.5, 0.3, 0.1, 1.0)
        # As T→∞, NS factors → 0, so raw → beta0
        val = c.evaluate(100.0)
        expected = math.tanh(0.5)
        assert val == pytest.approx(expected, abs=5e-3)

    def test_output_bounded(self) -> None:
        c = SabrAtmVolCurve(2.0, 1.0, 0.5, 0.5, alpha=0.0, beta_scale=1.0)
        for t in [0.1, 0.5, 1.0, 5.0, 10.0, 30.0]:
            val = c.evaluate(t)
            assert -1.0 < val < 1.0

    def test_negative_maturity(self) -> None:
        c = SabrAtmVolCurve(0.5, 0.3, 0.1, 1.0)
        assert c.evaluate(-1.0) == c.evaluate(0.0)

    def test_parameters_property(self) -> None:
        c = SabrAtmVolCurve(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    def test_callable_syntax(self) -> None:
        c = SabrAtmVolCurve(0.5, 0.3, 0.0, 1.0)
        assert c(1.0) == c.evaluate(1.0)


class TestSabrNuCurve:
    """Tests for SabrNuCurve."""

    def test_at_zero_maturity(self) -> None:
        c = SabrNuCurve(0.5, 0.3, 0.0, 1.0)
        expected = math.tanh(0.8)
        assert c.evaluate(0.0) == pytest.approx(expected)

    def test_at_large_maturity(self) -> None:
        c = SabrNuCurve(0.5, 0.3, 0.1, 1.0)
        val = c.evaluate(100.0)
        expected = math.tanh(0.5)
        assert val == pytest.approx(expected, abs=5e-3)

    def test_output_bounded(self) -> None:
        c = SabrNuCurve(2.0, 1.0, 0.5, 0.5)
        for t in [0.1, 1.0, 10.0]:
            assert -1.0 < c.evaluate(t) < 1.0

    def test_parameters_property(self) -> None:
        c = SabrNuCurve(0.1, 0.2, 0.3, 0.4)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4)


class TestSabrRhoCurve:
    """Tests for SabrRhoCurve."""

    def test_at_zero_maturity(self) -> None:
        c = SabrRhoCurve(0.0, -0.5, 0.0, 1.0)
        expected = math.tanh(-0.5)
        assert c.evaluate(0.0) == pytest.approx(expected)

    def test_at_large_maturity(self) -> None:
        c = SabrRhoCurve(-0.3, 0.1, 0.05, 1.0)
        val = c.evaluate(100.0)
        expected = math.tanh(-0.3)
        assert val == pytest.approx(expected, abs=5e-3)

    def test_correlation_bounded(self) -> None:
        c = SabrRhoCurve(1.0, -2.0, 0.5, 0.5)
        for t in [0.0, 0.5, 1.0, 5.0, 20.0]:
            val = c.evaluate(t)
            assert -1.0 < val < 1.0

    def test_parameters_property(self) -> None:
        c = SabrRhoCurve(0.1, 0.2, 0.3, 0.4)
        assert c.parameters == (0.1, 0.2, 0.3, 0.4)

    def test_imports_from_package(self) -> None:
        from hyesg.math.curves import SabrAtmVolCurve as Atm
        from hyesg.math.curves import SabrNuCurve as Nu
        from hyesg.math.curves import SabrRhoCurve as Rho

        assert Atm is not None
        assert Nu is not None
        assert Rho is not None
