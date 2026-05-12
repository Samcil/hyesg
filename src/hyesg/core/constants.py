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

# Gauss-Kronrod quadrature
GAUSS_KRONROD_BOUNDS_TOL: float = 1e-15
GAUSS_KRONROD_DEFAULT_TOL: float = 1e-8
GAUSS_KRONROD_MAX_DEPTH: int = 20

# Time consistency
TIME_CONSISTENCY_ROUND: float = 1e-15
TIME_CONSISTENCY_PLACES: int = 15

# Yield curve calibration
INITIAL_YC_TARGET_COINCIDENCE_TOL: float = 1e-5
AKIMA_CSV_ROUND_PLACES: int = 12

# Numerical limits
LIMIT_EPSILON: float = 1e-8

# LM optimizer defaults
LM_DEFAULT_MAX_ITER: int = 50
LM_DEFAULT_TOL: float = 1e-8
LM_DEFAULT_DAMPING: float = 0.01

# Bond pricing
BOND_YTM_TOL: float = 1e-10
BOND_YTM_MAX_ITER: int = 100

# Credit
CREDIT_RECOVERY_QUADRATURE_TOL: float = 1e-8

# Regime proportional ordering
REGIME_TRIAL_ORDERING_SEED_FACTOR: int = 1000003
REGIME_COPULA_SEED_OFFSET: int = 13
REGIME_CHI2_SEED_FACTOR: int = -104723
REGIME_CHI2_SEED_OFFSET: int = -1000003
