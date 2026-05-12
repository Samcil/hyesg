"""Configuration module: params, models, builder, templates, validation, funds."""

from hyesg.config.builder import SimulationBuilder
from hyesg.config.calibration_builder import CalibrationParametersBuilder
from hyesg.config.calibration_params import (
    CalibrationParameters,
    CIR2PPStructuralParams,
    CorrelationSpec,
    CreditCalibrationParams,
    EquityCalibrationParams,
    FXCalibrationParams,
    G2PPStructuralParams,
    RegimeDefinition,
    YieldCurveSpec,
)
from hyesg.config.fee_wrappers import FEE_WRAPPERS, get_fee_wrappers
from hyesg.config.fund_builder import FundBuilder
from hyesg.config.fund_catalogue import FundCatalogue, build_default_catalogue
from hyesg.config.funds import (
    FundCategory,
    FundDefinition,
    FundRebalanceStrategy,
    HoldingSpec,
    NetOfFeesFund,
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
    # fund catalogue
    "FEE_WRAPPERS",
    "FundBuilder",
    "FundCatalogue",
    "FundCategory",
    "FundDefinition",
    "FundRebalanceStrategy",
    "HoldingSpec",
    "NetOfFeesFund",
    "build_default_catalogue",
    "get_fee_wrappers",
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