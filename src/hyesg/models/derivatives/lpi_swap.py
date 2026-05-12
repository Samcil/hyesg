"""Limited Price Indexation (LPI) swap pricing.

LPI swaps pay inflation returns subject to annual caps and floors.
This module provides Monte Carlo pricing and equilibrium rate processing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class LPISwapConfig(NamedTuple):
    """LPI swap configuration.

    Attributes:
        cap: Annual inflation cap (e.g. 0.05 for 5%).
        floor: Annual inflation floor (e.g. 0.0 for 0%).
        maturity: Swap maturity in years.
        notional: Notional amount.
        payment_freq: Payment frequency (1 = annual).
    """

    cap: float = 0.05
    floor: float = 0.0
    maturity: float = 25.0
    notional: float = 1.0
    payment_freq: int = 1


class LPISwapPricer:
    """Price Limited Price Indexation (LPI) swaps.

    LPI = max(floor, min(cap, realised_inflation))

    The LPI swap pays the compounded capped-and-floored annual
    inflation rate over the life of the swap.
    """

    def capped_floored_inflation(
        self,
        inflation_rate: Array,
        cap: float,
        floor: float,
    ) -> Array:
        """Apply cap and floor to inflation rate.

        Args:
            inflation_rate: Raw annual inflation rates.
            cap: Upper bound on each year's inflation.
            floor: Lower bound on each year's inflation.

        Returns:
            Clipped inflation rates in [floor, cap].
        """
        return jnp.clip(inflation_rate, floor, cap)

    def price_lpi_swap(
        self,
        inflation_paths: Array,
        discount_factors: Array,
        config: LPISwapConfig,
    ) -> Array:
        """Price LPI swap via Monte Carlo.

        Computes the fair LPI swap rate by averaging the discounted
        capped-and-floored cumulative inflation across trials.

        Args:
            inflation_paths: Annual inflation rates, shape (n_trials, n_steps).
            discount_factors: Discount factors at each step, shape (n_steps,).
            config: LPI swap configuration.

        Returns:
            Fair LPI swap rate (scalar).
        """
        # Apply cap and floor to each year's inflation
        capped = self.capped_floored_inflation(
            inflation_paths, config.cap, config.floor
        )

        # Compound the capped inflation into an index
        # cumulative_index[i] = prod(1 + capped[0..i])
        cumulative_index = jnp.cumprod(1.0 + capped, axis=1)

        # Discounted terminal payoff: (cumulative_index - 1) * df
        n_steps = inflation_paths.shape[1]
        terminal_df = discount_factors[jnp.minimum(n_steps - 1, jnp.arange(n_steps))]
        terminal_df_last = terminal_df[-1]

        terminal_payoff = (cumulative_index[:, -1] - 1.0) * terminal_df_last
        fair_rate = jnp.mean(terminal_payoff) * config.notional

        return fair_rate


class EquilibriumSwapRateProcessor:
    """Post-process MC paths to equilibrium swap rates.

    Adjusts the Monte Carlo swap rate for liquidity premium
    and market conventions.

    Attributes:
        liquidity_premium: Additive liquidity premium adjustment.
    """

    def __init__(self, liquidity_premium: float = 0.0) -> None:
        """Initialise with liquidity premium.

        Args:
            liquidity_premium: Additive spread to adjust MC rate.
        """
        self.liquidity_premium = liquidity_premium

    def process(
        self,
        mc_swap_rate: Array,
        market_rate: float,
    ) -> Array:
        """Adjust MC rate to equilibrium.

        Adds the liquidity premium to the raw Monte Carlo swap rate.

        Args:
            mc_swap_rate: Raw Monte Carlo swap rate.
            market_rate: Observed market swap rate (for reference).

        Returns:
            Adjusted equilibrium swap rate.
        """
        return mc_swap_rate + self.liquidity_premium
