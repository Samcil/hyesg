"""Orchestration layer for hyesg simulations.

Provides batch execution, device management, and composable pipelines.

Public API::

    from hyesg.orchestration import (
        BatchResult,
        BatchRunner,
        DeviceInfo,
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
from hyesg.orchestration.device import DeviceInfo, detect_devices, select_device
from hyesg.orchestration.pipeline import Pipeline, PipelineContext
from hyesg.orchestration.protocols import PipelineStep, ProgressCallback
from hyesg.orchestration.steps import SimulateStep, TimingStep, ValidateStep

__all__ = [
    "BatchResult",
    "BatchRunner",
    "DeviceInfo",
    "Pipeline",
    "PipelineContext",
    "PipelineStep",
    "ProgressCallback",
    "SimulateStep",
    "TimingStep",
    "ValidateStep",
    "detect_devices",
    "select_device",
]
