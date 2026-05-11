"""Simulation configuration models for hyesg.

Pydantic v2 schemas that define the complete specification for
an ESG simulation run.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hyesg.config.params import CopulaType, PhiConfig  # noqa: TC001


class ModelConfig(BaseModel):
    """Configuration for a single model in the simulation.

    Attributes:
        type: Model type key (must match registry).
        name: Unique instance name.
        params: Model-specific parameters as a dict.
        dependencies: Names of models this depends on.
        phi: Optional shift function configuration.
        outputs: Output names to extract from this model.
        output_maturities: Maturities for term-structure outputs.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    phi: PhiConfig | None = None
    outputs: list[str] = Field(default_factory=list)
    output_maturities: list[float] = Field(default_factory=list)


class CorrelationEntry(BaseModel):
    """A single correlation between two shock streams.

    Attributes:
        shock_a: Name of the first shock.
        shock_b: Name of the second shock.
        value: Correlation coefficient in [-1, 1].
    """

    model_config = ConfigDict(frozen=True)

    shock_a: str
    shock_b: str
    value: float = Field(ge=-1.0, le=1.0)


class RegimeConfig(BaseModel):
    """Configuration for a simulation regime.

    Attributes:
        name: Regime identifier.
        n_trials: Number of Monte Carlo trials (must be > 0).
        seed: RNG seed for this regime.
        param_overrides: Per-model parameter overrides.
        blending_weights: Weights for regime blending.
        target_expectations: Target expectations for calibration.
        correlation_overrides: Per-regime correlation overrides.
        use_antithetic: Whether to use antithetic variates.
        copula: Copula type for shock generation.
        copula_df: Degrees of freedom for Student-t copula.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    n_trials: int = Field(default=1667, gt=0)
    seed: int = 27
    param_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    blending_weights: dict[str, float] | None = None
    target_expectations: dict[str, float] | None = None
    correlation_overrides: list[CorrelationEntry] = Field(default_factory=list)
    use_antithetic: bool = True
    copula: CopulaType | None = None
    copula_df: float = 5.0


class TimeGridConfig(BaseModel):
    """Time grid configuration for the simulation.

    Attributes:
        start_year: Simulation start time in years.
        end_year: Simulation end time in years.
        frequency: Time step frequency.
        custom_times: Custom time points (overrides frequency).
    """

    model_config = ConfigDict(frozen=True)

    start_year: float = 0.0
    end_year: float = 100.0
    frequency: Literal["monthly", "quarterly", "semi_annual", "annual"] = "monthly"
    custom_times: list[float] | None = None

    @field_validator("custom_times")
    @classmethod
    def _validate_custom_times(
        cls,
        v: list[float] | None,
    ) -> list[float] | None:
        """Validate custom times are strictly increasing."""
        if v is not None:
            if len(v) < 2:
                raise ValueError("custom_times must have at least 2 points")
            for i in range(1, len(v)):
                if v[i] <= v[i - 1]:
                    raise ValueError(
                        f"custom_times must be strictly increasing, "
                        f"but got {v[i]} <= {v[i - 1]} at index {i}"
                    )
        return v

    @property
    def time_points(self) -> jnp.ndarray:
        """Return time points as a JAX array."""
        if self.custom_times is not None:
            return jnp.array(self.custom_times)
        freq_map = {
            "monthly": 1.0 / 12.0,
            "quarterly": 0.25,
            "semi_annual": 0.5,
            "annual": 1.0,
        }
        dt = freq_map[self.frequency]
        n_steps = int(round((self.end_year - self.start_year) / dt))
        return jnp.linspace(self.start_year, self.end_year, n_steps + 1)

    @property
    def n_steps(self) -> int:
        """Number of timesteps (number of intervals)."""
        return int(len(self.time_points) - 1)

    @property
    def dt(self) -> float:
        """Nominal timestep size."""
        freq_map = {
            "monthly": 1.0 / 12.0,
            "quarterly": 0.25,
            "semi_annual": 0.5,
            "annual": 1.0,
        }
        if self.custom_times is not None:
            pts = self.custom_times
            return pts[1] - pts[0]
        return freq_map[self.frequency]


class PostProcessorConfig(BaseModel):
    """Configuration for a post-simulation processor.

    Attributes:
        type: Processor type identifier.
        params: Processor-specific parameters.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["sabr", "lsmc", "equilibrium_swap_rate"]
    params: dict[str, Any] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Complete simulation configuration.

    Attributes:
        name: Simulation name.
        description: Human-readable description.
        time_grid: Time grid configuration.
        models: List of model configurations.
        correlations: Correlation entries between shocks.
        regimes: List of regime configurations.
        post_processors: Post-processing steps.
        output_models: Names of models to include in output.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    time_grid: TimeGridConfig = Field(default_factory=TimeGridConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    correlations: list[CorrelationEntry] = Field(default_factory=list)
    regimes: list[RegimeConfig] = Field(default_factory=list)
    post_processors: list[PostProcessorConfig] = Field(default_factory=list)
    output_models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> SimulationConfig:
        """Validate internal cross-references."""
        model_names = {m.name for m in self.models}

        # Check dependencies reference existing models
        for model in self.models:
            for dep in model.dependencies:
                if dep not in model_names:
                    raise ValueError(
                        f"Model '{model.name}' depends on '{dep}' which is not defined"
                    )

        # Check output_models reference existing models
        for name in self.output_models:
            if name not in model_names:
                raise ValueError(
                    f"output_models references '{name}' which is not defined"
                )

        return self
