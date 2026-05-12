"""Tests for Gauss-Kronrod G7/K15 adaptive quadrature."""

from __future__ import annotations

import math

import pytest

from hyesg.math.quadrature import gauss_kronrod_integrate


def test_constant():
    """∫₀¹ 5 dx = 5."""
    assert gauss_kronrod_integrate(lambda x: 5.0, 0, 1) == pytest.approx(
        5.0, abs=1e-12
    )


def test_polynomial():
    """∫₀¹ x² dx = 1/3."""
    assert gauss_kronrod_integrate(lambda x: x**2, 0, 1) == pytest.approx(
        1 / 3, abs=1e-12
    )


def test_exponential():
    """∫₀¹ exp(x) dx = e - 1."""
    assert gauss_kronrod_integrate(math.exp, 0, 1) == pytest.approx(
        math.e - 1, abs=1e-10
    )


def test_sin():
    """∫₀^π sin(x) dx = 2."""
    assert gauss_kronrod_integrate(math.sin, 0, math.pi) == pytest.approx(
        2.0, abs=1e-10
    )


def test_oscillatory():
    """∫₀^{2π} sin(10x) dx = 0."""
    result = gauss_kronrod_integrate(
        lambda x: math.sin(10 * x), 0, 2 * math.pi, tolerance=1e-10
    )
    assert abs(result) < 1e-8


def test_bounds_collapse():
    """Zero-width interval returns 0."""
    assert gauss_kronrod_integrate(math.exp, 1.0, 1.0) == 0.0


def test_negative_interval():
    """∫₁⁰ f = -∫₀¹ f."""
    fwd = gauss_kronrod_integrate(lambda x: x**2, 0, 1)
    rev = gauss_kronrod_integrate(lambda x: x**2, 1, 0)
    assert rev == pytest.approx(-fwd, abs=1e-12)


def test_peaked_function():
    """Sharp peak requiring adaptive refinement.

    ∫₀¹ 1/(1 + 1000(x-0.5)²) dx = (2/√1000)·arctan(√1000/2).
    """
    exact = (2.0 / math.sqrt(1000)) * math.atan(math.sqrt(1000) / 2)
    result = gauss_kronrod_integrate(
        lambda x: 1.0 / (1.0 + 1000 * (x - 0.5) ** 2),
        0,
        1,
        tolerance=1e-8,
    )
    assert result == pytest.approx(exact, rel=1e-6)


def test_negative_tolerance_raises():
    """Negative tolerance raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        gauss_kronrod_integrate(lambda x: x, 0, 1, tolerance=-1)


def test_zero_tolerance_raises():
    """Zero tolerance raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        gauss_kronrod_integrate(lambda x: x, 0, 1, tolerance=0)
