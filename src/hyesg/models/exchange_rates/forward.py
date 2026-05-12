"""FX forward pricing via covered interest rate parity.

The FCA framework prices FX forwards using the relationship:

    F(t, T) = S(t) × P_foreign(t, T) / P_domestic(t, T)

where P_foreign and P_domestic are zero-coupon bond prices from
the respective yield curves, and S(t) is the spot FX rate.

Mark-to-market of an existing forward contract struck at rate K
with delivery at T, valued at time t:

    MTM(t) = P_domestic(t, T) × (F(t, T) − K)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

# ─── Protocols ───


@runtime_checkable
class FXForwardPricer(Protocol):
    """Protocol for FX forward pricing engines."""

    def forward_rate(
        self,
        spot: Array,
        domestic_zcb: Array,
        foreign_zcb: Array,
    ) -> Array:
        """Compute the FX forward rate via covered interest rate parity.

        Args:
            spot: Current spot FX rate S(t).
            domestic_zcb: Domestic ZCB price P_d(t, T).
            foreign_zcb: Foreign ZCB price P_f(t, T).

        Returns:
            Forward FX rate F(t, T).
        """
        ...

    def mark_to_market(
        self,
        spot: Array,
        domestic_zcb: Array,
        foreign_zcb: Array,
        strike: Array,
    ) -> Array:
        """Mark-to-market value of an existing forward contract.

        Args:
            spot: Current spot FX rate S(t).
            domestic_zcb: Domestic ZCB price P_d(t, T).
            foreign_zcb: Foreign ZCB price P_f(t, T).
            strike: Contracted forward rate K.

        Returns:
            MTM value from the perspective of the forward buyer.
        """
        ...


@runtime_checkable
class TransactionCostModel(Protocol):
    """Protocol for transaction cost models."""

    def cost(self, notional: Array) -> Array:
        """Compute the transaction cost for a given notional.

        Args:
            notional: Absolute notional traded.

        Returns:
            Transaction cost (positive value, to be subtracted from P&L).
        """
        ...


# ─── Forward Pricer Implementation ───


class FCAForwardPricer:
    """FX forward pricer using covered interest rate parity.

    Implements the FCA forward pricing relationship:

        F(t, T) = S(t) × P_f(t, T) / P_d(t, T)

    All methods are pure functions operating on JAX arrays,
    suitable for use inside ``jax.jit`` and ``jax.vmap``.
    """

    def forward_rate(
        self,
        spot: Array,
        domestic_zcb: Array,
        foreign_zcb: Array,
    ) -> Array:
        """Compute FX forward rate via covered interest rate parity.

        Args:
            spot: Current spot FX rate S(t).
            domestic_zcb: Domestic ZCB price P_d(t, T).
            foreign_zcb: Foreign ZCB price P_f(t, T).

        Returns:
            Forward FX rate F(t, T) = S(t) × P_f(t, T) / P_d(t, T).
        """
        return spot * foreign_zcb / jnp.maximum(domestic_zcb, 1e-15)

    def mark_to_market(
        self,
        spot: Array,
        domestic_zcb: Array,
        foreign_zcb: Array,
        strike: Array,
    ) -> Array:
        """Mark-to-market of an existing FX forward contract.

        MTM(t) = P_d(t, T) × (F(t, T) − K)

        where F(t, T) is the current forward rate and K is the strike.

        Args:
            spot: Current spot FX rate S(t).
            domestic_zcb: Domestic ZCB price P_d(t, T).
            foreign_zcb: Foreign ZCB price P_f(t, T).
            strike: Contracted forward rate K.

        Returns:
            MTM value (positive = in-the-money for buyer).
        """
        fwd = self.forward_rate(spot, domestic_zcb, foreign_zcb)
        return domestic_zcb * (fwd - strike)


# ─── Transaction Cost Implementation ───


class ConstantBidOfferSpread:
    """Constant bid-offer spread transaction cost model.

    Applies a fixed spread (in basis points) to the traded notional
    on each hedge roll or rebalance.

    Args:
        spread_bps: Half-spread in basis points (e.g. 5.0 = 5 bps).
            The full round-trip cost is 2 × spread_bps.
    """

    def __init__(self, spread_bps: float = 5.0) -> None:
        self._half_spread = spread_bps / 10_000.0

    @property
    def half_spread(self) -> float:
        """Half-spread as a decimal fraction."""
        return self._half_spread

    def cost(self, notional: Array) -> Array:
        """Transaction cost = |notional| × half_spread.

        Args:
            notional: Absolute notional traded.

        Returns:
            Transaction cost (always non-negative).
        """
        return jnp.abs(notional) * self._half_spread


# ─── FX Forward Model (registered, dep-aware) ───


class FXForward:
    """FX forward model integrated with the simulation engine.

    Computes forward FX rates and MTM values at each timestep
    by reading spot FX, domestic and foreign ZCB prices from deps.

    This model does NOT require its own shocks — it is purely
    derived from other model outputs.

    Args:
        name: Unique model name for registry.
        spot_fx_model: Dep key for the spot FX model.
        domestic_rate_model: Dep key for domestic rate model (provides zcb_price).
        foreign_rate_model: Dep key for foreign rate model (provides zcb_price).
        tenors: Forward tenors in years (e.g. [0.25, 0.5, 1.0]).
        pricer: FX forward pricer instance.
        transaction_cost: Optional transaction cost model.
    """

    def __init__(
        self,
        name: str = "fx_forward",
        spot_fx_model: str = "",
        domestic_rate_model: str = "",
        foreign_rate_model: str = "",
        tenors: tuple[float, ...] = (0.25, 0.5, 1.0),
        pricer: FXForwardPricer | None = None,
        transaction_cost: TransactionCostModel | None = None,
    ) -> None:
        self._name = name
        self._spot_fx = spot_fx_model
        self._domestic = domestic_rate_model
        self._foreign = foreign_rate_model
        self._tenors = tenors
        self._pricer = pricer or FCAForwardPricer()
        self._tc = transaction_cost

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def tenors(self) -> tuple[float, ...]:
        """Forward tenors in years."""
        return self._tenors

    def forward_rates(
        self,
        spot: Array,
        domestic_zcbs: dict[float, Array],
        foreign_zcbs: dict[float, Array],
    ) -> dict[float, Array]:
        """Compute forward rates for all configured tenors.

        Args:
            spot: Current spot FX rate.
            domestic_zcbs: Tenor → domestic ZCB price mapping.
            foreign_zcbs: Tenor → foreign ZCB price mapping.

        Returns:
            Tenor → forward FX rate mapping.
        """
        one = jnp.array(1.0, dtype=jnp.float64)
        result: dict[float, Array] = {}
        for tenor in self._tenors:
            d_zcb = domestic_zcbs.get(tenor, one)
            f_zcb = foreign_zcbs.get(tenor, one)
            result[tenor] = self._pricer.forward_rate(spot, d_zcb, f_zcb)
        return result

    def step(
        self,
        t: float,
        dt: float,
        deps: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute forward rates from dependency outputs.

        This is a stateless computation — no carry state required.

        Args:
            t: Current time in years.
            dt: Timestep size (unused, present for interface consistency).
            deps: Dependency outputs keyed by model name.

        Returns:
            Dict with keys ``"forward_{tenor}"`` for each tenor,
            plus ``"spot"`` echoing the spot rate.
        """
        one = jnp.array(1.0, dtype=jnp.float64)

        spot = (
            deps.get(self._spot_fx, {}).get("level", one)
            if self._spot_fx
            else one
        )

        outputs: dict[str, Any] = {"spot": spot}

        for tenor in self._tenors:
            # Read ZCB prices from deps
            d_zcb = (
                deps.get(self._domestic, {}).get(f"zcb_{tenor}", one)
                if self._domestic
                else one
            )
            f_zcb = (
                deps.get(self._foreign, {}).get(f"zcb_{tenor}", one)
                if self._foreign
                else one
            )
            fwd = self._pricer.forward_rate(spot, d_zcb, f_zcb)
            outputs[f"forward_{tenor}"] = fwd

        return outputs
