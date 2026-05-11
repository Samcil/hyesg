"""Numerical tolerances and constants.

Ported from Hymans.FinancialMaths.Constants.Tolerances.
"""

from __future__ import annotations

# Black implied vol solver
BLACK_IV_TOLERANCE: float = 1e-15

# Time comparison
TIME_EPSILON: float = 1e-7
TIME_PLACES: int = 7

# Compounding
COMPOUNDING_PRECISION: float = 1e-8

# Yield curve
YIELD_CURVE_EPSILON: float = 1e-12

# Parametric curve
PARAMETRIC_CURVE_EPSILON: float = 1e-8

# Volatility
VOLATILITY_PRECISION: float = 1e-8

# Numerical derivative
NUMERICAL_DERIVATIVE_H: float = 1e-4

# Portfolio optimisation (Brent's method)
PORTFOLIO_BRENT_TOL: float = 1e-8
PORTFOLIO_BRENT_ITERS: int = 100

# Portfolio thresholds
MIN_TRANSACTION: float = 1e-6
MIN_VALUE: float = 1e-8
SHORTING_TEST: float = 1e-12

# CIR-specific
CIR_ZERO_VOL_THRESHOLD: float = 1e-7
CIR_ZERO_H_THRESHOLD: float = 1e-8

# Default RNG seed
DEFAULT_RNG_SEED: int = 27
