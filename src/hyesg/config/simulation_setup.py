"""Top-level simulation setup and fluent builder.

Defines ``SimulationSetup`` (the complete simulation specification) and
``SimulationSetupBuilder`` (a fluent API for constructing one).  These
map to the C# ``SimulationSetup`` class used by the ESG engine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SetupRegimeConfig(BaseModel):
    """Configuration for a single regime within a SimulationSetup.

    Attributes:
        name: Human-readable regime label (e.g. ``"Strong"``).
        trials: Number of Monte Carlo trials for this regime.
        weight: Regime probability weight (computed from
            ``trials / total_trials``).
        calibration_params: Regime-specific calibration overrides.
    """

    model_config = ConfigDict(frozen=False)

    name: str
    trials: int
    weight: float = 0.0
    calibration_params: dict[str, Any] = Field(default_factory=dict)


class SimulationSetup(BaseModel):
    """Complete simulation configuration — the top-level object.

    This is the Python counterpart of the C# ``SimulationSetup`` class.
    It holds every parameter needed to run a full ESG simulation:
    time grid, regimes, economies, correlation matrix, fund catalogue,
    and post-processing recipe.

    Attributes:
        seed: Master RNG seed.
        horizon: Projection horizon in years.
        inverse_dt: Number of time steps per year (e.g. 12 = monthly).
        regimes: List of regime configurations.
        economies: Economy objects from ``config/economy.py``.
        correlation: Correlation matrix (``SymmetricGeneralizedMatrix``).
        funds: Fund catalogue.
        post_processing: Post-processing recipe.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int = 27
    horizon: int = 100
    inverse_dt: int = 12
    regimes: list[SetupRegimeConfig] = Field(default_factory=list)
    economies: list[Any] = Field(default_factory=list)
    correlation: Any | None = None
    funds: Any | None = None
    post_processing: Any | None = None

    @property
    def n_steps(self) -> int:
        """Total number of time steps (``horizon × inverse_dt``)."""
        return self.horizon * self.inverse_dt

    @property
    def dt(self) -> float:
        """Length of a single time step in years."""
        return 1.0 / self.inverse_dt

    @property
    def total_trials(self) -> int:
        """Sum of trial counts across all regimes."""
        return sum(r.trials for r in self.regimes)

    def validate_setup(self) -> list[str]:
        """Validate cross-references and structural invariants.

        Returns:
            List of error messages.  Empty list means valid.
        """
        errors: list[str] = []
        if not self.regimes:
            errors.append("No regimes defined")
        if not self.economies:
            errors.append("No economies defined")
        if self.total_trials == 0:
            errors.append("Total trials is 0")
        return errors


class SimulationSetupBuilder:
    """Fluent builder for :class:`SimulationSetup`.

    Usage::

        setup = (
            SimulationSetupBuilder()
            .seed(27)
            .time_grid(horizon=100, inverse_dt=12)
            .add_regime("Strong", trials=2500)
            .add_regime("Moderate", trials=1500)
            .add_regime("Weak", trials=1000)
            .add_economy(gbp_economy)
            .build()
        )
    """

    def __init__(self) -> None:
        self._seed: int = 27
        self._horizon: int = 100
        self._inverse_dt: int = 12
        self._regimes: list[SetupRegimeConfig] = []
        self._economies: list[Any] = []
        self._correlation: Any = None
        self._funds: Any = None
        self._post_processing: Any = None

    def seed(self, s: int) -> SimulationSetupBuilder:
        """Set the master RNG seed."""
        self._seed = s
        return self

    def time_grid(
        self, horizon: int = 100, inverse_dt: int = 12
    ) -> SimulationSetupBuilder:
        """Configure the time grid.

        Args:
            horizon: Projection horizon in years.
            inverse_dt: Steps per year (12 = monthly).
        """
        self._horizon = horizon
        self._inverse_dt = inverse_dt
        return self

    def add_regime(
        self, name: str, trials: int, **params: Any
    ) -> SimulationSetupBuilder:
        """Add a regime to the simulation.

        Args:
            name: Regime label.
            trials: Number of trials for this regime.
            **params: Additional calibration parameter overrides.
        """
        self._regimes.append(
            SetupRegimeConfig(
                name=name,
                trials=trials,
                calibration_params=params,
            )
        )
        return self

    def add_economy(self, economy: Any) -> SimulationSetupBuilder:
        """Add an economy to the simulation."""
        self._economies.append(economy)
        return self

    def correlate(self, matrix: Any) -> SimulationSetupBuilder:
        """Set the correlation matrix."""
        self._correlation = matrix
        return self

    def add_fund_catalogue(self, catalogue: Any) -> SimulationSetupBuilder:
        """Set the fund catalogue."""
        self._funds = catalogue
        return self

    def post_processing(self, recipe: Any) -> SimulationSetupBuilder:
        """Set the post-processing recipe."""
        self._post_processing = recipe
        return self

    def build(self, *, validate: bool = True) -> SimulationSetup:
        """Build and optionally validate the simulation setup.

        Computes regime weights from trial counts.  When *validate* is
        ``True`` (default), runs :meth:`SimulationSetup.validate_setup`
        and raises on errors.  Pass ``validate=False`` for partial
        setups (e.g. when economies will be added later).

        Args:
            validate: Whether to run full validation.

        Returns:
            A :class:`SimulationSetup` instance.

        Raises:
            ValueError: If *validate* is ``True`` and the setup is invalid.
        """
        setup = SimulationSetup(
            seed=self._seed,
            horizon=self._horizon,
            inverse_dt=self._inverse_dt,
            regimes=self._regimes,
            economies=self._economies,
            correlation=self._correlation,
            funds=self._funds,
            post_processing=self._post_processing,
        )

        # Compute regime weights from trial proportions
        total = setup.total_trials
        for regime in setup.regimes:
            regime.weight = regime.trials / total if total > 0 else 0.0

        if validate:
            errors = setup.validate_setup()
            if errors:
                raise ValueError(f"Invalid setup: {errors}")

        return setup
