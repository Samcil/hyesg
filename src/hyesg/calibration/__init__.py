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
    RobustLevenbergMarquardt,
    RobustLevenbergMarquardtConfig,
    ScipyMinimize,
)
from hyesg.calibration.protocols import CalibrationDataReader, ObjectiveFunction, Optimizer
from hyesg.calibration.result import CalibrationResult, OptimizationResult
from hyesg.calibration.sabr import (
    SabrCalibrator,
    SabrTermStructure,
    nelson_siegel_tanh,
    sabr_implied_vol_hagan,
)
from hyesg.calibration.yield_curve_config import (
    LongEndExtensionConfig,
    RpiReformConfig,
    YieldCurvePipelineConfig,
)
from hyesg.calibration.yield_curve_model import InitialYieldCurveModel
from hyesg.calibration.yield_curves import (
    YIELD_CURVE_KNOTS,
    PowerBlend,
    YieldCurveCalibrationResult,
    breakeven_cpi_forward_curve,
    build_akima_yield_curve,
    build_forward_rate_curve,
    build_real_yield_curve,
    calibrate_yield_curve,
    fisher_real_rate,
    reform_adjusted_forward_curve,
)

__all__ = [
    # Protocols
    "CalibrationDataReader",
    "ObjectiveFunction",
    "Optimizer",
    # Results
    "CalibrationResult",
    "OptimizationResult",
    # Optimizers
    "LevenbergMarquardt",
    "LevenbergMarquardtConfig",
    "RobustLevenbergMarquardt",
    "RobustLevenbergMarquardtConfig",
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
    # Yield Curves — pipeline
    "YIELD_CURVE_KNOTS",
    "YieldCurveCalibrationResult",
    "build_akima_yield_curve",
    "build_forward_rate_curve",
    "build_real_yield_curve",
    "calibrate_yield_curve",
    "fisher_real_rate",
    # Yield Curves — RPI reform / CPI
    "PowerBlend",
    "breakeven_cpi_forward_curve",
    "reform_adjusted_forward_curve",
    # Yield Curves — config and model
    "InitialYieldCurveModel",
    "LongEndExtensionConfig",
    "RpiReformConfig",
    "YieldCurvePipelineConfig",
    # SABR calibration
    "SabrCalibrator",
    "SabrTermStructure",
    "nelson_siegel_tanh",
    "sabr_implied_vol_hagan",
]
