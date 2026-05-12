"""Calibration framework for hyesg models.

Provides optimizers, objective functions, and a high-level calibrator
for fitting model parameters to market data.
"""

from __future__ import annotations

from hyesg.calibration.calibrator import Calibrator
from hyesg.calibration.objectives import (
    cir_curve_objective,
    cir_curve_objective_direct,
    credit_spread_objective,
    credit_spread_objective_direct,
    ou_curve_objective,
    ou_curve_objective_direct,
    ou_zcb_price,
)
from hyesg.calibration.optimizer import (
    LevenbergMarquardt,
    LevenbergMarquardtConfig,
    ScipyMinimize,
)
from hyesg.calibration.protocols import ObjectiveFunction, Optimizer
from hyesg.calibration.result import CalibrationResult, OptimizationResult
from hyesg.calibration.yield_curves import (
    YIELD_CURVE_KNOTS,
    YieldCurveCalibrationResult,
    build_akima_yield_curve,
    build_real_yield_curve,
    calibrate_yield_curve,
    fisher_real_rate,
)

__all__ = [
    # Protocols
    "ObjectiveFunction",
    "Optimizer",
    # Results
    "CalibrationResult",
    "OptimizationResult",
    # Optimizers
    "LevenbergMarquardt",
    "LevenbergMarquardtConfig",
    "ScipyMinimize",
    # Objectives
    "cir_curve_objective",
    "cir_curve_objective_direct",
    "credit_spread_objective",
    "credit_spread_objective_direct",
    "ou_curve_objective",
    "ou_curve_objective_direct",
    "ou_zcb_price",
    # Calibrator
    "Calibrator",
    # Yield Curves
    "YIELD_CURVE_KNOTS",
    "YieldCurveCalibrationResult",
    "build_akima_yield_curve",
    "build_real_yield_curve",
    "calibrate_yield_curve",
    "fisher_real_rate",
]
