"""Fluent builder for simulation configuration.

Provides a convenient API for constructing SimulationConfig
objects programmatically.
"""

from __future__ import annotations

from typing import Any

from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)


class SimulationBuilder:
    """Fluent builder for SimulationConfig.

    Example::

        config = (
            SimulationBuilder("my_sim")
            .time_grid(0.0, 50.0, "monthly")
            .add_model("cir2pp", "nominal", alpha1=0.1, mu1=0.03)
            .add_model("g2pp", "real", alpha1=0.05, mu1=0.01)
            .correlate("nominal_z1", "real_z1", 0.3)
            .add_regime("base", n_trials=5000, seed=42)
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        """Initialise builder with simulation name.

        Args:
            name: Simulation name.
        """
        self._name = name
        self._description = ""
        self._time_grid = TimeGridConfig()
        self._models: list[ModelConfig] = []
        self._correlations: list[CorrelationEntry] = []
        self._regimes: list[RegimeConfig] = []
        self._output_models: list[str] = []

    def description(self, desc: str) -> SimulationBuilder:
        """Set the simulation description.

        Args:
            desc: Human-readable description.

        Returns:
            Self for chaining.
        """
        self._description = desc
        return self

    def time_grid(
        self,
        start: float = 0.0,
        end: float = 100.0,
        frequency: str = "monthly",
    ) -> SimulationBuilder:
        """Configure the time grid.

        Args:
            start: Start year.
            end: End year.
            frequency: Step frequency.

        Returns:
            Self for chaining.
        """
        self._time_grid = TimeGridConfig(
            start_year=start,
            end_year=end,
            frequency=frequency,  # type: ignore[arg-type]
        )
        return self

    def add_model(
        self,
        model_type: str,
        name: str,
        *,
        dependencies: list[str] | None = None,
        outputs: list[str] | None = None,
        output_maturities: list[float] | None = None,
        **params: Any,
    ) -> SimulationBuilder:
        """Add a model to the simulation.

        Args:
            model_type: Model type key (registry name).
            name: Unique instance name.
            dependencies: Names of dependent models.
            outputs: Output names to extract.
            output_maturities: Maturities for outputs.
            **params: Model-specific parameters.

        Returns:
            Self for chaining.
        """
        self._models.append(
            ModelConfig(
                type=model_type,
                name=name,
                params=params,
                dependencies=dependencies or [],
                outputs=outputs or [],
                output_maturities=output_maturities or [],
            )
        )
        return self

    def correlate(self, shock_a: str, shock_b: str, value: float) -> SimulationBuilder:
        """Add a correlation between two shock streams.

        Args:
            shock_a: First shock name.
            shock_b: Second shock name.
            value: Correlation coefficient [-1, 1].

        Returns:
            Self for chaining.
        """
        self._correlations.append(
            CorrelationEntry(
                shock_a=shock_a,
                shock_b=shock_b,
                value=value,
            )
        )
        return self

    def add_regime(
        self,
        name: str,
        *,
        n_trials: int = 1667,
        seed: int = 27,
        **kwargs: Any,
    ) -> SimulationBuilder:
        """Add a simulation regime.

        Args:
            name: Regime name.
            n_trials: Number of trials.
            seed: RNG seed.
            **kwargs: Additional regime parameters.

        Returns:
            Self for chaining.
        """
        self._regimes.append(
            RegimeConfig(
                name=name,
                n_trials=n_trials,
                seed=seed,
                **kwargs,
            )
        )
        return self

    def output_models(self, *names: str) -> SimulationBuilder:
        """Specify which models to include in output.

        Args:
            *names: Model names.

        Returns:
            Self for chaining.
        """
        self._output_models.extend(names)
        return self

    def build(self) -> SimulationConfig:
        """Build and validate the SimulationConfig.

        Returns:
            Validated SimulationConfig.

        Raises:
            ValidationError: If configuration is invalid.
        """
        return SimulationConfig(
            name=self._name,
            description=self._description,
            time_grid=self._time_grid,
            models=self._models,
            correlations=self._correlations,
            regimes=self._regimes,
            output_models=self._output_models,
        )
