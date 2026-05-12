"""Orchestration layer for hyesg simulations.

Provides batch execution, device management, composable pipelines,
and correlation matrix assembly from CSV blocks.

Public API::

    from hyesg.orchestration import (
        BatchResult,
        BatchRunner,
        CorrelationAssembler,
        CreditCorrelationConfig,
        DeviceInfo,
        DzFactorLabelRegistry,
        Pipeline,
        PipelineContext,
        PipelineStep,
        ProgressCallback,
        SimulateStep,
        TimingStep,
        ValidateStep,
        detect_devices,
        select_device,
    )
"""

from __future__ import annotations

from hyesg.orchestration.batch import BatchResult, BatchRunner
from hyesg.orchestration.correlation_assembler import (
    CorrelationAssembler,
    CreditCorrelationConfig,
)
from hyesg.orchestration.device import DeviceInfo, detect_devices, select_device
from hyesg.orchestration.label_registry import DzFactorLabelRegistry
from hyesg.orchestration.pipeline import Pipeline, PipelineContext
from hyesg.orchestration.protocols import PipelineStep, ProgressCallback
from hyesg.orchestration.steps import (
    CalibrationStep,
    PostProcessStep,
    SimulateStep,
    TimingStep,
    ValidateStep,
)

__all__ = [
    "BatchResult",
    "BatchRunner",
    "CalibrationStep",
    "CorrelationAssembler",
    "CreditCorrelationConfig",
    "DeviceInfo",
    "DzFactorLabelRegistry",
    "Pipeline",
    "PipelineContext",
    "PipelineStep",
    "PostProcessStep",
    "ProgressCallback",
    "SimulateStep",
    "TimingStep",
    "ValidateStep",
    "detect_devices",
    "select_device",
]
