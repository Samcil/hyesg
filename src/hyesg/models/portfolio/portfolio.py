"""Portfolio aggregation — post-processing of simulation results.

This module is NOT a stochastic model. It does not participate in the
``jax.lax.scan`` simulation loop. Instead it takes a ``SimulationResult``
and aggregates asset-level outputs into portfolio-level returns, values,
and weight histories.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from hyesg.models.portfolio.analytics import PortfolioAnalytics
from hyesg.models.portfolio.result import PortfolioConfig, PortfolioResult
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.engine.output import SimulationResult

logger = logging.getLogger(__name__)

# Default output field used when extracting asset returns
_DEFAULT_RETURN_FIELD = OutputName.LOG_RETURN


class Portfolio:
    """Aggregate asset-level simulation results into a portfolio.

    Args:
        config: Portfolio configuration (weights, rebalancing strategy, etc.).
        return_field: Name of the output field to use as asset returns.
            Defaults to ``"log_return"``.
    """

    def __init__(
        self,
        config: PortfolioConfig,
        return_field: str = _DEFAULT_RETURN_FIELD,
    ) -> None:
        self._config = config
        self._return_field = return_field

    @property
    def config(self) -> PortfolioConfig:
        """The portfolio configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, result: SimulationResult) -> PortfolioResult:
        """Aggregate simulation results into portfolio-level outputs.

        Args:
            result: A ``SimulationResult`` containing per-asset outputs.

        Returns:
            A ``PortfolioResult`` with portfolio returns, values and
            optional weight history.

        Raises:
            KeyError: If a required model or field is missing from *result*.
        """
        asset_names = list(self._config.weights.keys())
        weights_list = [self._config.weights[name] for name in asset_names]
        weights = jnp.asarray(weights_list)

        # Warn if weights don't sum to ~1.0
        total_weight = jnp.sum(weights)
        if not jnp.isclose(total_weight, 1.0, atol=1e-6):
            warnings.warn(
                f"Portfolio weights sum to {float(total_weight):.6f}, expected ~1.0",
                stacklevel=2,
            )

        # Extract asset returns: list of arrays each (n_trials, n_steps)
        asset_returns = self._extract_asset_returns(result, asset_names)

        # Stack to (n_trials, n_steps, n_assets)
        returns_stack = jnp.stack(asset_returns, axis=-1)

        # Apply currency adjustment if configured
        if self._config.currency_base is not None:
            returns_stack = self._apply_currency_adjustment(
                result, returns_stack, asset_names
            )

        # Compute portfolio returns and weight history
        if self._config.rebalance == "constant_mix":
            port_returns = _constant_mix_returns(returns_stack, weights)
            weights_history = None
        elif self._config.rebalance == "buy_and_hold":
            port_returns, weights_history = _buy_and_hold_returns(
                returns_stack, weights
            )
        else:  # periodic
            port_returns, weights_history = _periodic_rebalance_returns(
                returns_stack, weights, self._config.rebalance_frequency
            )

        # Portfolio value path
        values = PortfolioAnalytics.portfolio_value(
            self._config.initial_value, port_returns
        )

        return PortfolioResult(
            returns=port_returns,
            values=values,
            weights_history=weights_history,
            config=self._config,
            asset_names=asset_names,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_asset_returns(
        self, result: SimulationResult, asset_names: list[str]
    ) -> list[Array]:
        """Extract per-asset return arrays from the simulation result."""
        arrays: list[Array] = []
        for name in asset_names:
            arrays.append(result.select(name, self._return_field))
        return arrays

    def _apply_currency_adjustment(
        self,
        result: SimulationResult,
        returns_stack: Array,
        asset_names: list[str],
    ) -> Array:
        """Add FX returns for foreign assets (simplified unhedged).

        For each asset whose name differs from the currency base, look up
        an FX model named ``fx_{asset_name}`` and add its log return.
        """
        for idx, name in enumerate(asset_names):
            if name == self._config.currency_base:
                continue
            fx_model = f"fx_{name}"
            try:
                fx_return = result.select(fx_model, self._return_field)
                adjusted = returns_stack[:, :, idx] + fx_return
                returns_stack = returns_stack.at[:, :, idx].set(adjusted)
            except KeyError:
                logger.debug(
                    "No FX model '%s' found; treating '%s' as domestic",
                    fx_model,
                    name,
                )
        return returns_stack


# ======================================================================
# Pure functions — all JIT-compatible
# ======================================================================


def _constant_mix_returns(returns_stack: Array, weights: Array) -> Array:
    """Compute constant-mix portfolio returns.

    At every step the portfolio is rebalanced to the target weights.

    Args:
        returns_stack: Asset returns with shape (n_trials, n_steps, n_assets).
        weights: Target weights with shape (n_assets,).

    Returns:
        Portfolio returns with shape (n_trials, n_steps).
    """
    return jnp.sum(returns_stack * weights, axis=-1)


def _buy_and_hold_returns(
    returns_stack: Array, initial_weights: Array
) -> tuple[Array, Array]:
    """Compute buy-and-hold portfolio returns with drifting weights.

    Weights evolve with cumulative asset returns; no rebalancing.

    Args:
        returns_stack: **Arithmetic** asset returns with shape
            ``(n_trials, n_steps, n_assets)``.  If the simulation
            produces log returns, convert first via
            ``jnp.exp(log_returns) - 1``.
        initial_weights: Starting weights with shape ``(n_assets,)``.

    Returns:
        Tuple of ``(portfolio_returns, weights_history)`` where
        ``portfolio_returns`` has shape ``(n_trials, n_steps)`` and
        ``weights_history`` has shape ``(n_trials, n_steps, n_assets)``.
    """
    n_trials, n_steps, n_assets = returns_stack.shape

    # Cumulative growth factors per asset: (n_trials, n_steps, n_assets)
    # NOTE: returns_stack must contain arithmetic returns, not log returns.
    # If the simulation produces log returns, the caller must convert via
    # ``jnp.exp(log_returns) - 1`` before passing to this function.
    growth = jnp.cumprod(1.0 + returns_stack, axis=1)

    # Value of each asset at each step (unnormalised)
    # At step 0 we use growth from step 0 only
    asset_values = initial_weights * growth

    # Total portfolio value at each step
    port_total = jnp.sum(asset_values, axis=-1, keepdims=True)

    # Drifting weights — guard against total wipeout (0/0 → NaN)
    safe_total = jnp.maximum(port_total, 1e-30)
    weights_history = asset_values / safe_total

    # Portfolio returns: weighted by *beginning-of-period* weights
    # At step 0: weights = initial_weights
    bop_weights = jnp.concatenate(
        [
            jnp.broadcast_to(initial_weights, (n_trials, 1, n_assets)),
            weights_history[:, :-1, :],
        ],
        axis=1,
    )

    port_returns = jnp.sum(returns_stack * bop_weights, axis=-1)

    return port_returns, weights_history


def _periodic_rebalance_returns(
    returns_stack: Array, target_weights: Array, frequency: int
) -> tuple[Array, Array]:
    """Compute portfolio returns with periodic rebalancing.

    Between rebalances weights drift (buy-and-hold); at rebalance dates
    weights reset to the target.  Uses ``jax.lax.scan`` so the function
    is JIT-safe.

    Args:
        returns_stack: Arithmetic asset returns, shape
            ``(n_trials, n_steps, n_assets)``.
        target_weights: Target weights with shape ``(n_assets,)``.
        frequency: Steps between rebalances.

    Returns:
        Tuple of ``(portfolio_returns, weights_history)`` where
        ``portfolio_returns`` has shape ``(n_trials, n_steps)`` and
        ``weights_history`` has shape ``(n_trials, n_steps, n_assets)``.
    """
    n_trials, n_steps, n_assets = returns_stack.shape

    step_indices = jnp.arange(n_steps)
    # Next step is a rebalance point → reset weights after this step
    is_next_rebal = jnp.concatenate(
        [((step_indices[1:] % frequency) == 0), jnp.array([False])]
    )

    def scan_fn(
        current_weights: Array,
        inputs: tuple[Array, Array],
    ) -> tuple[Array, tuple[Array, Array]]:
        step_returns, rebal_flag = inputs
        # Record beginning-of-period weights and compute portfolio return
        step_return = jnp.sum(step_returns * current_weights, axis=-1)

        # Drift weights based on asset growth
        asset_growth = current_weights * (1.0 + step_returns)
        total = jnp.sum(asset_growth, axis=-1, keepdims=True)
        drifted = asset_growth / jnp.maximum(total, 1e-30)

        # Rebalance or drift
        next_weights = jnp.where(
            rebal_flag,
            jnp.broadcast_to(target_weights, current_weights.shape),
            drifted,
        )
        return next_weights, (step_return, current_weights)

    init_weights = jnp.broadcast_to(
        target_weights, (n_trials, n_assets)
    ).copy()

    # Transpose returns for scan: (n_steps, n_trials, n_assets)
    returns_t = jnp.transpose(returns_stack, (1, 0, 2))

    _, (port_returns, weights_history) = jax.lax.scan(
        scan_fn, init_weights, (returns_t, is_next_rebal)
    )

    # Transpose back: (n_trials, n_steps)
    port_returns = jnp.transpose(port_returns, (1, 0))
    weights_history = jnp.transpose(weights_history, (1, 0, 2))

    return port_returns, weights_history
