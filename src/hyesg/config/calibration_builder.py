"""Fluent builder for CalibrationParameters.

Provides a convenient API for constructing CalibrationParameters
objects programmatically with method chaining.
"""

from __future__ import annotations

from typing import Any

from hyesg.config.calibration_params import (
    CalibrationParameters,
    CorrelationSpec,
    CreditCalibrationParams,
    EquityCalibrationParams,
    FXCalibrationParams,
    RegimeDefinition,
    YieldCurveSpec,
)


class CalibrationParametersBuilder:
    """Fluent builder for CalibrationParameters.

    Example::

        params = (
            CalibrationParametersBuilder()
            .with_seed(42)
            .with_horizon(50, inverse_dt=12)
            .with_trials(5000)
            .with_regime("Strong", trials=2500, weight=0.5)
            .with_equity("FTSE100", dividend_yield=0.03, volatility=0.18)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialise builder with default values."""
        self._seed: int = 27
        self._inverse_dt: int = 12
        self._horizon: int = 100
        self._trials: int = 5000
        self._regimes: list[RegimeDefinition] = []
        self._nominal_curves: dict[str, YieldCurveSpec] = {}
        self._real_curves: dict[str, YieldCurveSpec] = {}
        self._inflation_targets: dict[str, float] = {}
        self._equity_params: dict[str, EquityCalibrationParams] = {}
        self._fx_params: dict[str, FXCalibrationParams] = {}
        self._credit_params: dict[str, CreditCalibrationParams] = {}
        self._correlation_specs: dict[str, CorrelationSpec] = {}
        self._cir2pp_structural: Any = None
        self._g2pp_structural: Any = None

    def with_seed(self, seed: int) -> CalibrationParametersBuilder:
        """Set the master RNG seed.

        Args:
            seed: Random number generator seed.

        Returns:
            Self for chaining.
        """
        self._seed = seed
        return self

    def with_horizon(
        self,
        horizon: int,
        inverse_dt: int = 12,
    ) -> CalibrationParametersBuilder:
        """Set projection horizon and time step frequency.

        Args:
            horizon: Projection horizon in years.
            inverse_dt: Time steps per year.

        Returns:
            Self for chaining.
        """
        self._horizon = horizon
        self._inverse_dt = inverse_dt
        return self

    def with_trials(self, trials: int) -> CalibrationParametersBuilder:
        """Set total number of Monte Carlo trials.

        Args:
            trials: Number of trials.

        Returns:
            Self for chaining.
        """
        self._trials = trials
        return self

    def with_regime(
        self,
        name: str,
        trials: int,
        weight: float = 0.5,
        **overrides: Any,
    ) -> CalibrationParametersBuilder:
        """Add a regime definition.

        Args:
            name: Regime name.
            trials: Trials for this regime.
            weight: Blending weight.
            **overrides: Parameter overrides for this regime.

        Returns:
            Self for chaining.
        """
        self._regimes.append(
            RegimeDefinition(
                name=name,
                trials=trials,
                weight=weight,
                overrides=overrides,
            )
        )
        return self

    def with_nominal_curve(
        self,
        name: str,
        spec: YieldCurveSpec,
    ) -> CalibrationParametersBuilder:
        """Add a nominal yield curve.

        Args:
            name: Currency or curve identifier.
            spec: Yield curve specification.

        Returns:
            Self for chaining.
        """
        self._nominal_curves[name] = spec
        return self

    def with_real_curve(
        self,
        name: str,
        spec: YieldCurveSpec,
    ) -> CalibrationParametersBuilder:
        """Add a real yield curve.

        Args:
            name: Currency or curve identifier.
            spec: Yield curve specification.

        Returns:
            Self for chaining.
        """
        self._real_curves[name] = spec
        return self

    def with_equity(
        self,
        name: str,
        **params: Any,
    ) -> CalibrationParametersBuilder:
        """Add equity calibration parameters.

        Args:
            name: Equity index identifier.
            **params: Parameters passed to EquityCalibrationParams.

        Returns:
            Self for chaining.
        """
        self._equity_params[name] = EquityCalibrationParams(**params)
        return self

    def with_fx(
        self,
        name: str,
        **params: Any,
    ) -> CalibrationParametersBuilder:
        """Add FX pair calibration parameters.

        Args:
            name: FX pair identifier.
            **params: Parameters passed to FXCalibrationParams.

        Returns:
            Self for chaining.
        """
        self._fx_params[name] = FXCalibrationParams(**params)
        return self

    def with_credit(
        self,
        name: str,
        **params: Any,
    ) -> CalibrationParametersBuilder:
        """Add credit class calibration parameters.

        Args:
            name: Credit class identifier.
            **params: Parameters passed to CreditCalibrationParams.

        Returns:
            Self for chaining.
        """
        self._credit_params[name] = CreditCalibrationParams(**params)
        return self

    def with_correlation(
        self,
        name: str,
        spec: CorrelationSpec,
    ) -> CalibrationParametersBuilder:
        """Add a correlation matrix specification.

        Args:
            name: Correlation group identifier.
            spec: Correlation specification.

        Returns:
            Self for chaining.
        """
        self._correlation_specs[name] = spec
        return self

    def build(self) -> CalibrationParameters:
        """Build and validate the CalibrationParameters.

        Returns:
            Validated CalibrationParameters instance.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        return CalibrationParameters(
            seed=self._seed,
            inverse_dt=self._inverse_dt,
            horizon=self._horizon,
            trials=self._trials,
            regimes=tuple(self._regimes),
            nominal_curves=self._nominal_curves,
            real_curves=self._real_curves,
            inflation_targets=self._inflation_targets,
            equity_params=self._equity_params,
            fx_params=self._fx_params,
            credit_params=self._credit_params,
            correlation_specs=self._correlation_specs,
            cir2pp_structural=self._cir2pp_structural,
            g2pp_structural=self._g2pp_structural,
        )
