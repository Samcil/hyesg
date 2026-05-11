"""Portfolio analytics — pure-function statistics on portfolio arrays.

All functions operate on JAX arrays and are JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class PortfolioAnalytics:
    """Static analytics for portfolio returns and values.

    Every method is a pure function operating on JAX arrays.
    Returns are assumed to have shape (n_trials, n_steps) and
    values have shape (n_trials, n_steps + 1).
    """

    @staticmethod
    def cumulative_return(returns: Array) -> Array:
        """Cumulative return from period returns.

        Args:
            returns: Period returns with shape (n_trials, n_steps).

        Returns:
            Cumulative returns with shape (n_trials, n_steps).
        """
        return jnp.cumprod(1.0 + returns, axis=1) - 1.0

    @staticmethod
    def portfolio_value(initial: float, returns: Array) -> Array:
        """Portfolio value path from an initial value and period returns.

        Args:
            initial: Starting portfolio value.
            returns: Period returns with shape (n_trials, n_steps).

        Returns:
            Value path with shape (n_trials, n_steps + 1) where the first
            column is the initial value.
        """
        growth = jnp.cumprod(1.0 + returns, axis=1)
        ones = jnp.ones((returns.shape[0], 1))
        return initial * jnp.concatenate([ones, growth], axis=1)

    @staticmethod
    def drawdown(values: Array) -> Array:
        """Drawdown at each point relative to the running maximum.

        Args:
            values: Portfolio values with shape (n_trials, n_steps + 1).

        Returns:
            Drawdown array with the same shape as *values*. Values are
            non-negative fractions (0 = no drawdown, 1 = total loss).
        """
        running_max = jnp.maximum.accumulate(values, axis=1)
        return jnp.where(running_max > 0, (running_max - values) / running_max, 0.0)

    @staticmethod
    def max_drawdown(values: Array) -> Array:
        """Maximum drawdown per trial.

        Args:
            values: Portfolio values with shape (n_trials, n_steps + 1).

        Returns:
            Maximum drawdown per trial with shape (n_trials,).
        """
        dd = PortfolioAnalytics.drawdown(values)
        return jnp.max(dd, axis=1)

    @staticmethod
    def sharpe_ratio(returns: Array, risk_free: float | Array = 0.0) -> Array:
        """Sharpe ratio per trial (not annualised).

        Args:
            returns: Period returns with shape (n_trials, n_steps).
            risk_free: Risk-free rate (scalar or broadcastable array).

        Returns:
            Sharpe ratio per trial with shape (n_trials,).
        """
        excess = returns - risk_free
        return jnp.mean(excess, axis=1) / jnp.std(excess, axis=1)

    @staticmethod
    def volatility(returns: Array) -> Array:
        """Volatility (standard deviation) of returns per trial.

        Args:
            returns: Period returns with shape (n_trials, n_steps).

        Returns:
            Volatility per trial with shape (n_trials,).
        """
        return jnp.std(returns, axis=1)
