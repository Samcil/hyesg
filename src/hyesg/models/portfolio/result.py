"""Result containers and configuration for portfolio aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


@dataclass
class PortfolioConfig:
    """Configuration for portfolio aggregation.

    Attributes:
        weights: Mapping of model_name -> weight. Weights should sum to ~1.0.
        rebalance: Rebalancing strategy. One of "buy_and_hold", "periodic",
            or "constant_mix".
        rebalance_frequency: Number of steps between rebalances (for periodic).
        currency_base: If set, apply currency adjustment for foreign assets.
        initial_value: Starting portfolio value.
    """

    weights: dict[str, float]
    rebalance: str = "buy_and_hold"
    rebalance_frequency: int = 1
    currency_base: str | None = None
    initial_value: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if not self.weights:
            raise ValueError("weights must not be empty")
        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for '{name}' is {weight}; all weights must be >= 0"
                )
        valid_strategies = {"buy_and_hold", "periodic", "constant_mix"}
        if self.rebalance not in valid_strategies:
            raise ValueError(
                f"rebalance must be one of {valid_strategies}, got '{self.rebalance}'"
            )
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be >= 1")


@dataclass
class PortfolioResult:
    """Container for portfolio aggregation results.

    Attributes:
        returns: Portfolio returns with shape (n_trials, n_steps).
        values: Portfolio value path with shape (n_trials, n_steps + 1).
        weights_history: Drifting weights with shape
            (n_trials, n_steps, n_assets), or None for constant-mix.
        config: The PortfolioConfig used to produce this result.
        asset_names: Ordered list of asset model names.
    """

    returns: Array
    values: Array
    weights_history: Array | None
    config: PortfolioConfig
    asset_names: list[str] = field(default_factory=list)
