"""Shared fixtures for orchestration tests."""

from __future__ import annotations

import pytest

import hyesg.models  # noqa: F401 — populate model registry
from hyesg.config.models import (
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import CIRParams
from hyesg.models import CIR


@pytest.fixture()
def cir_config() -> SimulationConfig:
    """A minimal CIR simulation config (10 trials, 12 annual steps)."""
    return SimulationConfig(
        name="test_cir",
        time_grid=TimeGridConfig(
            start_year=0.0,
            end_year=12.0,
            frequency="annual",
        ),
        models=[ModelConfig(type="cir", name="nominal")],
        regimes=[RegimeConfig(name="r1", n_trials=10, seed=42)],
    )


@pytest.fixture()
def cir_model() -> CIR:
    """A CIR model instance for testing."""
    return CIR(
        params=CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05),
        name="nominal",
    )


@pytest.fixture()
def cir_config_with_model(
    cir_config: SimulationConfig,
    cir_model: CIR,
) -> tuple[SimulationConfig, dict[str, CIR]]:
    """Config + pre-built model dict (avoids registry lookup)."""
    return cir_config, {"nominal": cir_model}
