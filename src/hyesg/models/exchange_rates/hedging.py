"""Currency hedging via rolling FX forward contracts.

Implements the ``CurrencyHedgedEquityRebalancer`` pattern from the
C# ESG engine.  At each rebalance date the hedger:

1. Settles the expiring forward contract (P&L = F_prev − S_now).
2. Enters a new forward at the current market forward rate.
3. Applies transaction costs via a ``TransactionCostModel``.

The hedged equity return decomposes as:

    R_hedged = R_local_equity + (1 − h) × R_fx + h × R_hedge

where h is the hedge ratio, R_fx is the unhedged FX return, and
R_hedge is the forward premium/discount.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from hyesg.models.exchange_rates.forward import (
    ConstantBidOfferSpread,
    FCAForwardPricer,
    FXForwardPricer,
    TransactionCostModel,
)


@runtime_checkable
class CurrencyHedger(Protocol):
    """Protocol for currency hedging strategies."""

    def hedge_return(
        self,
        equity_return_local: Array,
        fx_return: Array,
        forward_premium: Array,
    ) -> Array:
        """Compute the hedged return for one period.

        Args:
            equity_return_local: Local-currency equity return.
            fx_return: Spot FX return over the period.
            forward_premium: Forward premium (F/S − 1) for the period.

        Returns:
            Hedged total return in domestic currency.
        """
        ...


# ─── Hedge State ───


class HedgeState:
    """Mutable state for a rolling hedge position.

    Tracks the current forward strike, accumulated P&L, and
    time until next roll.

    Args:
        forward_strike: Current forward contract strike rate.
        periods_to_roll: Periods remaining until next roll.
        cum_hedge_pnl: Cumulative hedge P&L.
        cum_transaction_costs: Cumulative transaction costs.
    """

    __slots__ = (
        "forward_strike",
        "periods_to_roll",
        "cum_hedge_pnl",
        "cum_transaction_costs",
    )

    def __init__(
        self,
        forward_strike: float = 0.0,
        periods_to_roll: int = 0,
        cum_hedge_pnl: float = 0.0,
        cum_transaction_costs: float = 0.0,
    ) -> None:
        self.forward_strike = forward_strike
        self.periods_to_roll = periods_to_roll
        self.cum_hedge_pnl = cum_hedge_pnl
        self.cum_transaction_costs = cum_transaction_costs


# ─── Currency Hedged Equity Rebalancer ───


class CurrencyHedgedEquityRebalancer:
    """Rolling FX forward hedge for foreign equity exposures.

    Implements periodic hedging by entering FX forward contracts
    that are rolled at the configured frequency.  The hedge ratio
    controls the fraction of FX exposure that is hedged.

    P&L decomposition per period:

        unhedged_return = equity_local × (1 + fx_return) − 1
        hedge_gain = h × (forward_premium − fx_return)
        transaction_cost = tc.cost(notional)
        hedged_return = unhedged_return + hedge_gain − transaction_cost

    Args:
        hedge_ratio: Fraction of FX exposure hedged (0.0 to 1.0).
        roll_frequency_months: Forward tenor / roll frequency in months.
        pricer: FX forward pricer (defaults to ``FCAForwardPricer``).
        transaction_cost: Transaction cost model (defaults to zero-cost).
    """

    def __init__(
        self,
        hedge_ratio: float = 1.0,
        roll_frequency_months: int = 12,
        pricer: FXForwardPricer | None = None,
        transaction_cost: TransactionCostModel | None = None,
    ) -> None:
        if not 0.0 <= hedge_ratio <= 1.0:
            raise ValueError(
                f"hedge_ratio must be in [0, 1], got {hedge_ratio}"
            )
        if roll_frequency_months not in (1, 3, 6, 12):
            raise ValueError(
                f"roll_frequency_months must be 1, 3, 6, or 12, "
                f"got {roll_frequency_months}"
            )
        self._hedge_ratio = hedge_ratio
        self._roll_months = roll_frequency_months
        self._roll_tenor = roll_frequency_months / 12.0
        self._pricer = pricer or FCAForwardPricer()
        self._tc = transaction_cost or ConstantBidOfferSpread(spread_bps=0.0)

    @property
    def hedge_ratio(self) -> float:
        """Fraction of FX exposure hedged."""
        return self._hedge_ratio

    @property
    def roll_tenor(self) -> float:
        """Forward tenor in years."""
        return self._roll_tenor

    def hedge_return(
        self,
        equity_return_local: Array,
        fx_return: Array,
        forward_premium: Array,
    ) -> Array:
        """Compute hedged equity return for one period.

        Args:
            equity_return_local: Local-currency equity return.
            fx_return: Spot FX return (S_new/S_old − 1).
            forward_premium: Forward premium (F/S − 1).

        Returns:
            Hedged return in domestic currency.
        """
        # Unhedged return: local equity return converted at spot
        unhedged = (1.0 + equity_return_local) * (1.0 + fx_return) - 1.0

        # Hedge gain: difference between forward premium and realised FX
        hedge_gain = self._hedge_ratio * (forward_premium - fx_return)

        return unhedged + hedge_gain

    def compute_pnl_decomposition(
        self,
        equity_return_local: Array,
        fx_return: Array,
        forward_premium: Array,
        notional: Array,
    ) -> dict[str, Array]:
        """Full P&L decomposition for one period.

        Args:
            equity_return_local: Local-currency equity return.
            fx_return: Spot FX return (S_new/S_old − 1).
            forward_premium: Forward premium (F/S − 1).
            notional: Hedge notional for transaction cost calculation.

        Returns:
            Dict with keys:
                - ``unhedged_return``: Return without any hedging.
                - ``hedge_gain``: Gain from the forward hedge.
                - ``transaction_cost``: Cost of rolling/rebalancing.
                - ``hedged_return``: Net hedged return after costs.
        """
        unhedged = (1.0 + equity_return_local) * (1.0 + fx_return) - 1.0
        hedge_gain = self._hedge_ratio * (forward_premium - fx_return)
        tc = self._tc.cost(notional * self._hedge_ratio)

        hedged = unhedged + hedge_gain - tc

        return {
            "unhedged_return": unhedged,
            "hedge_gain": hedge_gain,
            "transaction_cost": tc,
            "hedged_return": hedged,
        }

    def should_roll(self, step_index: int, steps_per_year: int) -> bool:
        """Determine if the hedge should roll at this step.

        Args:
            step_index: Current simulation step (0-based).
            steps_per_year: Number of steps per year (e.g. 12 for monthly).

        Returns:
            True if the forward should be rolled at this step.
        """
        steps_per_roll = int(self._roll_months * steps_per_year / 12)
        if steps_per_roll < 1:
            return True
        return step_index % steps_per_roll == 0

    def compute_forward_premium(
        self,
        spot: Array,
        domestic_zcb: Array,
        foreign_zcb: Array,
    ) -> Array:
        """Forward premium as a fraction of spot.

        premium = F(t, T) / S(t) − 1 = P_f(t, T) / P_d(t, T) − 1

        Args:
            spot: Current spot FX rate (unused in premium calc, but
                  required by the pricer interface).
            domestic_zcb: Domestic ZCB price P_d(t, T).
            foreign_zcb: Foreign ZCB price P_f(t, T).

        Returns:
            Forward premium as a decimal fraction.
        """
        fwd = self._pricer.forward_rate(spot, domestic_zcb, foreign_zcb)
        return fwd / jnp.maximum(spot, 1e-15) - 1.0
