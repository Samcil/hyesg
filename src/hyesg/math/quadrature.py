"""Gauss-Kronrod G7/K15 adaptive quadrature.

Implements the 7-point Gauss and 15-point Kronrod rule for numerical
integration, with recursive bisection for error control. This matches
the C# implementation in Hymans.FinancialMaths.NumericalMethods.Integration.

The non-standard error estimate from C#:
    ε = 200 · |I_G7 - I_K15|^1.5

is used instead of the standard |I_G7 - I_K15| to provide more aggressive
refinement.
"""

from __future__ import annotations

from collections.abc import Callable

# ── Kronrod 15-point nodes and weights on [-1, 1] ──────────────────────

_K15_NODES: tuple[float, ...] = (
    -0.991455371120813,
    -0.949107912342759,
    -0.864864423359769,
    -0.741531185599394,
    -0.586087235467691,
    -0.405845151377397,
    -0.207784955007898,
    0.0,
    0.207784955007898,
    0.405845151377397,
    0.586087235467691,
    0.741531185599394,
    0.864864423359769,
    0.949107912342759,
    0.991455371120813,
)

_K15_WEIGHTS: tuple[float, ...] = (
    0.022935322010529,
    0.063092092629979,
    0.104790010322250,
    0.140653259715525,
    0.169004726639268,
    0.190350578064785,
    0.204432940075298,
    0.209482141084728,
    0.204432940075298,
    0.190350578064785,
    0.169004726639268,
    0.140653259715525,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
)

# ── Gauss 7-point weights (nodes are a subset of K15) ─────────────────
# G7 nodes correspond to K15 indices {1, 3, 5, 7, 9, 11, 13}.

_G7_INDICES: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13)

_G7_WEIGHTS: tuple[float, ...] = (
    0.129484966168870,
    0.279705391489277,
    0.381830050505119,
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
)


def _gauss_kronrod_step(
    f: Callable[[float], float],
    a: float,
    b: float,
) -> tuple[float, float]:
    """Single-interval G7/K15 evaluation.

    Args:
        f: Integrand function.
        a: Lower bound.
        b: Upper bound.

    Returns:
        (gauss_estimate, kronrod_estimate) for the interval [a, b].
    """
    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)

    # Evaluate f at all 15 Kronrod nodes (mapped to [a, b]).
    fvals = tuple(f(midpoint + half_width * t) for t in _K15_NODES)

    kronrod = sum(w * fv for w, fv in zip(_K15_WEIGHTS, fvals)) * half_width
    gauss = (
        sum(w * fvals[i] for w, i in zip(_G7_WEIGHTS, _G7_INDICES))
        * half_width
    )
    return gauss, kronrod


def _adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float,
    depth: int,
    max_depth: int,
) -> float:
    """Recursive adaptive bisection.

    Args:
        f: Integrand.
        a: Lower bound.
        b: Upper bound.
        tolerance: Absolute error tolerance.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.

    Returns:
        Approximate integral of f over [a, b].
    """
    if abs(b - a) < 1e-15:
        return 0.0

    gauss_est, kronrod_est = _gauss_kronrod_step(f, a, b)

    # Non-standard C# error estimate: 200 * |I_G - I_K|^1.5
    error = 200.0 * abs(gauss_est - kronrod_est) ** 1.5

    if error <= tolerance or depth >= max_depth:
        return kronrod_est

    mid = 0.5 * (a + b)
    left = _adaptive(f, a, mid, tolerance * 0.5, depth + 1, max_depth)
    right = _adaptive(f, mid, b, tolerance * 0.5, depth + 1, max_depth)
    return left + right


def gauss_kronrod_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float = 1e-8,
    max_depth: int = 20,
) -> float:
    """Adaptive Gauss-Kronrod G7/K15 integration.

    Integrates f over [a, b] using adaptive bisection with the
    15-point Kronrod rule and 7-point Gauss rule for error estimation.

    The non-standard error estimate ε = 200·|I_G - I_K|^1.5 is used,
    matching the C# ESG engine implementation.

    Args:
        f: Integrand function f: ℝ → ℝ.
        a: Lower integration bound.
        b: Upper integration bound.
        tolerance: Absolute error tolerance (default 1e-8).
        max_depth: Maximum recursion depth (default 20).

    Returns:
        Approximate value of ∫ₐᵇ f(x)dx.

    Raises:
        ValueError: If tolerance <= 0.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    # Handle reversed bounds: ∫ₐᵇ = -∫ᵦₐ
    if a > b:
        return -gauss_kronrod_integrate(f, b, a, tolerance, max_depth)

    return _adaptive(f, a, b, tolerance, depth=0, max_depth=max_depth)
