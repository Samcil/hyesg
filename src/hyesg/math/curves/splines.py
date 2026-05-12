"""Spline interpolation curves."""

from __future__ import annotations

from hyesg.math.curves.protocol import ParametricCurve


class CubicSpline(ParametricCurve):
    """Natural cubic spline interpolation.

    Computes a piecewise cubic polynomial that passes through all
    knot points with continuous first and second derivatives.
    Uses natural boundary conditions (second derivative = 0 at
    endpoints). For x outside the data range, uses flat
    extrapolation (returns endpoint values).

    Args:
        xs: Sorted array of x coordinates (knots).
        ys: Array of y coordinates at knots.
    """

    def __init__(self, xs: list[float], ys: list[float]) -> None:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) < 2:
            raise ValueError("Need at least 2 data points")

        self._xs = list(xs)
        self._ys = list(ys)
        self._n = len(xs)
        self._a, self._b, self._c, self._d = self._compute_coefficients()

    def _compute_coefficients(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Compute natural cubic spline coefficients.

        Returns:
            Tuple of (a, b, c, d) coefficient lists.
        """
        n = self._n
        xs = self._xs
        ys = self._ys

        a = list(ys)
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]

        # Solve tridiagonal system for c coefficients
        # Natural spline: c[0] = c[n-1] = 0
        alpha = [0.0] * n
        for i in range(1, n - 1):
            alpha[i] = 3.0 / h[i] * (a[i + 1] - a[i]) - 3.0 / h[i - 1] * (
                a[i] - a[i - 1]
            )

        # Thomas algorithm for tridiagonal system
        c = [0.0] * n
        diag = [1.0] * n
        mu = [0.0] * n
        z = [0.0] * n

        for i in range(1, n - 1):
            diag[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / diag[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / diag[i]

        # Back substitution
        for j in range(n - 2, 0, -1):
            c[j] = z[j] - mu[j] * c[j + 1]

        # Compute b and d from a and c
        b = [0.0] * (n - 1)
        d = [0.0] * (n - 1)
        for i in range(n - 1):
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

        return a, b, c, d

    def _find_interval(self, x: float) -> int:
        """Find the interval index for a given x using binary search.

        Args:
            x: The input value.

        Returns:
            Index i such that xs[i] <= x < xs[i+1].
        """
        lo, hi = 0, self._n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self._xs[mid]:
                hi = mid - 1
            elif x > self._xs[mid + 1]:
                lo = mid + 1
            else:
                return mid
        return lo

    def evaluate(self, x: float) -> float:
        """Evaluate the cubic spline at point x.

        Uses flat extrapolation outside the data range.

        Args:
            x: The input value.

        Returns:
            Interpolated value at x.
        """
        if x <= self._xs[0]:
            return self._ys[0]
        if x >= self._xs[-1]:
            return self._ys[-1]

        i = self._find_interval(x)
        dx = x - self._xs[i]
        return self._a[i] + self._b[i] * dx + self._c[i] * dx**2 + self._d[i] * dx**3

    def integral(self, a: float, b: float) -> float:
        """Exact integral of piecewise cubic over [a, b].

        Handles flat extrapolation outside knot range analytically.
        """
        return _piecewise_cubic_integral(
            self._xs, self._ys, self._a, self._b, self._c, self._d, a, b
        )


class AkimaCubicSpline(ParametricCurve):
    """Akima's modified cubic spline interpolation.

    Uses local weights to reduce oscillation near outliers.
    Same interface as CubicSpline but with Akima's slope
    estimation algorithm.

    Args:
        xs: Sorted array of x coordinates (knots).
        ys: Array of y coordinates at knots.
    """

    def __init__(self, xs: list[float], ys: list[float]) -> None:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) < 2:
            raise ValueError("Need at least 2 data points")

        self._xs = list(xs)
        self._ys = list(ys)
        self._n = len(xs)
        self._b, self._c, self._d = self._compute_coefficients()

    def _compute_coefficients(
        self,
    ) -> tuple[list[float], list[float], list[float]]:
        """Compute Akima spline coefficients.

        Returns:
            Tuple of (b, c, d) coefficient lists for each interval.
        """
        n = self._n
        xs = self._xs
        ys = self._ys

        # Compute divided differences
        m = [0.0] * (n + 3)
        for i in range(n - 1):
            m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])

        # Extrapolate slopes at boundaries
        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        m[n + 1] = 2.0 * m[n] - m[n - 1]
        m[n + 2] = 2.0 * m[n + 1] - m[n]

        # Compute slopes at each knot using Akima's weights
        slopes = [0.0] * n
        for i in range(n):
            w1 = abs(m[i + 3] - m[i + 2])
            w2 = abs(m[i + 1] - m[i])
            if w1 + w2 > 1e-30:
                slopes[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)
            else:
                slopes[i] = (m[i + 1] + m[i + 2]) / 2.0

        # Compute polynomial coefficients for each interval
        b = [0.0] * (n - 1)
        c = [0.0] * (n - 1)
        d = [0.0] * (n - 1)
        for i in range(n - 1):
            h = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            b[i] = slopes[i]
            c[i] = (3.0 * dy / h - 2.0 * slopes[i] - slopes[i + 1]) / h
            d[i] = (slopes[i] + slopes[i + 1] - 2.0 * dy / h) / (h * h)

        return b, c, d

    def _find_interval(self, x: float) -> int:
        """Find the interval index for a given x.

        Args:
            x: The input value.

        Returns:
            Index i such that xs[i] <= x < xs[i+1].
        """
        lo, hi = 0, self._n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self._xs[mid]:
                hi = mid - 1
            elif x > self._xs[mid + 1]:
                lo = mid + 1
            else:
                return mid
        return lo

    def evaluate(self, x: float) -> float:
        """Evaluate the Akima spline at point x.

        Uses flat extrapolation outside the data range.

        Args:
            x: The input value.

        Returns:
            Interpolated value at x.
        """
        if x <= self._xs[0]:
            return self._ys[0]
        if x >= self._xs[-1]:
            return self._ys[-1]

        i = self._find_interval(x)
        dx = x - self._xs[i]
        return self._ys[i] + self._b[i] * dx + self._c[i] * dx**2 + self._d[i] * dx**3

    def integral(self, a: float, b: float) -> float:
        """Exact integral of piecewise cubic over [a, b].

        Handles flat extrapolation outside knot range analytically.
        """
        return _piecewise_cubic_integral(
            self._xs, self._ys, self._ys, self._b, self._c, self._d, a, b
        )


def _segment_integral(a_coeff: float, b_coeff: float, c_coeff: float,
                       d_coeff: float, dx: float) -> float:
    """Antiderivative of a + b·t + c·t² + d·t³ evaluated at t=dx."""
    return a_coeff * dx + b_coeff * dx**2 / 2 + c_coeff * dx**3 / 3 + d_coeff * dx**4 / 4


def _piecewise_cubic_integral(
    xs: list[float],
    ys: list[float],
    a_coeffs: list[float],
    b_coeffs: list[float],
    c_coeffs: list[float],
    d_coeffs: list[float],
    lower: float,
    upper: float,
) -> float:
    """Exact integral of a piecewise cubic spline with flat extrapolation.

    The spline is f(x) = a[i] + b[i]·dx + c[i]·dx² + d[i]·dx³
    on each interval [xs[i], xs[i+1]], with flat extrapolation outside
    [xs[0], xs[-1]].

    Args:
        xs: Knot positions.
        ys: Values at knots (for flat extrapolation).
        a_coeffs: Constant coefficients per interval.
        b_coeffs: Linear coefficients per interval.
        c_coeffs: Quadratic coefficients per interval.
        d_coeffs: Cubic coefficients per interval.
        lower: Lower integration bound.
        upper: Upper integration bound.

    Returns:
        Exact value of ∫_lower^upper f(x) dx.
    """
    if upper < lower:
        return -_piecewise_cubic_integral(
            xs, ys, a_coeffs, b_coeffs, c_coeffs, d_coeffs, upper, lower
        )
    if upper == lower:
        return 0.0

    n = len(xs)
    x_lo = xs[0]
    x_hi = xs[-1]
    result = 0.0

    # Left flat extrapolation region: f(x) = ys[0] for x < xs[0]
    if lower < x_lo:
        right = min(upper, x_lo)
        result += ys[0] * (right - lower)
        if upper <= x_lo:
            return result
        lower = x_lo

    # Right flat extrapolation region: f(x) = ys[-1] for x > xs[-1]
    if upper > x_hi:
        left = max(lower, x_hi)
        result += ys[-1] * (upper - left)
        if lower >= x_hi:
            return result
        upper = x_hi

    # Interior: integrate over spline segments
    # Find starting segment
    i_start = 0
    for i in range(n - 2):
        if lower < xs[i + 1]:
            i_start = i
            break
    else:
        i_start = n - 2

    for i in range(i_start, n - 1):
        seg_lo = xs[i]
        seg_hi = xs[i + 1]

        # Clip to [lower, upper]
        lo = max(lower, seg_lo)
        hi = min(upper, seg_hi)
        if lo >= hi:
            continue

        # Integrate a[i] + b[i]·(x-xs[i]) + c[i]·(x-xs[i])² + d[i]·(x-xs[i])³
        dx_lo = lo - seg_lo
        dx_hi = hi - seg_lo
        result += (
            _segment_integral(a_coeffs[i], b_coeffs[i], c_coeffs[i], d_coeffs[i], dx_hi)
            - _segment_integral(a_coeffs[i], b_coeffs[i], c_coeffs[i], d_coeffs[i], dx_lo)
        )

        if hi >= upper:
            break

    return result
