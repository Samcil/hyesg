"""Configuration module: params, models, builder, templates, validation."""

from hyesg.config.builder import SimulationBuilder
from hyesg.config.calibration_builder import CalibrationParametersBuilder
from hyesg.config.calibration_params import (
    CIR2PPStructuralParams,
    CalibrationParameters,
    CorrelationSpec,
    CreditCalibrationParams,
    EquityCalibrationParams,
    FXCalibrationParams,
    G2PPStructuralParams,
    RegimeDefinition,
    YieldCurveSpec,
)
from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    PostProcessorConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import (
    CIRParams,
    CopulaType,
    GBMParams,
    OUParams,
    PhiConfig,
    RebalanceStrategy,
    RecoveryType,
)
from hyesg.config.templates import base_ess_template
from hyesg.config.validation import (
    ConfigValidationError,
    build_dep_graph,
    find_cycle_path,
    has_cycles,
    validate_config,
)

__all__ = [
    # calibration builder
    "CalibrationParametersBuilder",
    # calibration params
    "CIR2PPStructuralParams",
    "CalibrationParameters",
    "CorrelationSpec",
    "CreditCalibrationParams",
    "EquityCalibrationParams",
    "FXCalibrationParams",
    "G2PPStructuralParams",
    "RegimeDefinition",
    "YieldCurveSpec",
    # builder
    "SimulationBuilder",
    # models
    "CorrelationEntry",
    "ModelConfig",
    "PostProcessorConfig",
    "RegimeConfig",
    "SimulationConfig",
    "TimeGridConfig",
    # params
    "CIRParams",
    "CopulaType",
    "GBMParams",
    "OUParams",
    "PhiConfig",
    "RebalanceStrategy",
    "RecoveryType",
    # templates
    "base_ess_template",
    # validation
    "ConfigValidationError",
    "build_dep_graph",
    "find_cycle_path",
    "has_cycles",
    "validate_config",
]
