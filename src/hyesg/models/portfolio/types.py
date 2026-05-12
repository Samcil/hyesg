"""Portfolio type hierarchy — protocols and concrete implementations.

Provides a ``Portfolio`` protocol and concrete portfolio classes for
equity, bond, derivative, fund, currency-hedged, and composite
portfolios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from hyesg.models.portfolio.holdings import (
        BondHolding,
        CDSHolding,
        EquityHolding,
        ForwardHolding,
        SwapHolding,
    )
    from hyesg.models.portfolio.rebalancer import AllocationRebalancer


@runtime_checkable
class PortfolioProtocol(Protocol):
    """Protocol for portfolio types.

    Any object implementing ``value`` and ``return_`` can be used
    wherever a portfolio is expected.
    """

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Portfolio value at time *t* given simulation state.

        Args:
            state: Mapping of asset label to price/value array.
            t: Current time in years.

        Returns:
            Portfolio value as a JAX scalar.
        """
        ...

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Portfolio return over ``[t, t+dt]``.

        Args:
            state: Mapping of asset label to price/value array.
            t: Start of period.
            dt: Period length.

        Returns:
            Portfolio return as a JAX scalar.
        """
        ...


# ─── Equity Portfolio ───


class EquityPortfolio:
    """Weighted basket of equity holdings.

    The portfolio value is the weighted sum of individual equity prices,
    and the return is the weighted average of individual equity returns.

    Args:
        holdings: List of ``EquityHolding`` positions.
    """

    def __init__(self, holdings: list[EquityHolding]) -> None:
        self._holdings = holdings
        total_w = sum(h.weight for h in holdings)
        self._weights = {h.asset_label: h.weight / total_w for h in holdings}
        self._initial_prices = {h.asset_label: h.initial_price for h in holdings}

    @property
    def weights(self) -> dict[str, float]:
        """Normalised asset weights."""
        return dict(self._weights)

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Weighted portfolio value.

        Args:
            state: Asset label → current price array.
            t: Current time (unused for equities).

        Returns:
            Portfolio value.
        """
        val = jnp.asarray(0.0, dtype=jnp.float64)
        for label, w in self._weights.items():
            price = state.get(label, jnp.asarray(self._initial_prices[label]))
            init_p = self._initial_prices[label]
            val = val + w * (jnp.asarray(price, dtype=jnp.float64) / init_p)
        return val

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Weighted portfolio return.

        Expects ``state`` to contain keys ``"{label}_return"`` for each
        holding.

        Args:
            state: Asset label → return array (keyed as ``{label}_return``).
            t: Period start.
            dt: Period length.

        Returns:
            Weighted portfolio return.
        """
        ret = jnp.asarray(0.0, dtype=jnp.float64)
        for label, w in self._weights.items():
            asset_ret = state.get(
                f"{label}_return", jnp.asarray(0.0, dtype=jnp.float64)
            )
            ret = ret + w * jnp.asarray(asset_ret, dtype=jnp.float64)
        return ret


# ─── Bond Portfolio ───


class BondPortfolio:
    """Weighted basket of bond holdings with periodic rebalancing.

    Args:
        holdings: List of ``BondHolding`` positions.
        rebalancer: An ``AllocationRebalancer`` for periodic rebalancing.
    """

    def __init__(
        self,
        holdings: list[BondHolding],
        rebalancer: AllocationRebalancer,
    ) -> None:
        self._holdings = holdings
        self._rebalancer = rebalancer
        total_w = sum(h.weight for h in holdings)
        self._weights = {h.asset_label: h.weight / total_w for h in holdings}

    @property
    def weights(self) -> dict[str, float]:
        """Normalised asset weights."""
        return dict(self._weights)

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Weighted bond portfolio value.

        Args:
            state: Asset label → bond price array.
            t: Current time.

        Returns:
            Portfolio value.
        """
        val = jnp.asarray(0.0, dtype=jnp.float64)
        for label, w in self._weights.items():
            price = state.get(label, jnp.asarray(0.0, dtype=jnp.float64))
            val = val + w * jnp.asarray(price, dtype=jnp.float64)
        return val

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Weighted bond portfolio return.

        Args:
            state: Asset label → return array (keyed as ``{label}_return``).
            t: Period start.
            dt: Period length.

        Returns:
            Weighted portfolio return.
        """
        ret = jnp.asarray(0.0, dtype=jnp.float64)
        for label, w in self._weights.items():
            asset_ret = state.get(
                f"{label}_return", jnp.asarray(0.0, dtype=jnp.float64)
            )
            ret = ret + w * jnp.asarray(asset_ret, dtype=jnp.float64)
        return ret


# ─── Derivative Portfolio ───


class DerivativePortfolio:
    """Portfolio of swaps, CDS, and forwards.

    A simplified container that values derivative positions from
    simulation state.

    Args:
        swaps: List of ``SwapHolding`` positions.
        cds: List of ``CDSHolding`` positions.
        forwards: List of ``ForwardHolding`` positions.
    """

    def __init__(
        self,
        swaps: list[SwapHolding] | None = None,
        cds: list[CDSHolding] | None = None,
        forwards: list[ForwardHolding] | None = None,
    ) -> None:
        self._swaps = swaps or []
        self._cds = cds or []
        self._forwards = forwards or []

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Total derivative portfolio value.

        Args:
            state: Simulation state with derivative valuations.
            t: Current time.

        Returns:
            Sum of derivative MTM values.
        """
        val = jnp.asarray(0.0, dtype=jnp.float64)
        for fwd in self._forwards:
            fwd_val = state.get(
                f"{fwd.asset_label}_fwd", jnp.asarray(0.0, dtype=jnp.float64)
            )
            val = val + fwd.weight * jnp.asarray(fwd_val, dtype=jnp.float64)
        return val

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Derivative portfolio return.

        Args:
            state: Simulation state.
            t: Period start.
            dt: Period length.

        Returns:
            Portfolio return.
        """
        return jnp.asarray(0.0, dtype=jnp.float64)


# ─── Fund Portfolio ───


class FundPortfolio:
    """Recursive composition referencing other portfolios.

    A fund-of-funds that delegates to sub-portfolios identified by
    reference labels.

    Args:
        fund_refs: List of ``(fund_ref, weight)`` pairs.
        sub_portfolios: Mapping of fund_ref → ``PortfolioProtocol``.
    """

    def __init__(
        self,
        fund_refs: list[tuple[str, float]],
        sub_portfolios: dict[str, PortfolioProtocol],
    ) -> None:
        self._fund_refs = fund_refs
        self._sub_portfolios = sub_portfolios

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Weighted fund-of-funds value.

        Args:
            state: Simulation state.
            t: Current time.

        Returns:
            Weighted sum of sub-portfolio values.
        """
        val = jnp.asarray(0.0, dtype=jnp.float64)
        for ref, w in self._fund_refs:
            sub = self._sub_portfolios.get(ref)
            if sub is not None:
                val = val + w * sub.value(state, t)
        return val

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Weighted fund-of-funds return.

        Args:
            state: Simulation state.
            t: Period start.
            dt: Period length.

        Returns:
            Weighted sum of sub-portfolio returns.
        """
        ret = jnp.asarray(0.0, dtype=jnp.float64)
        for ref, w in self._fund_refs:
            sub = self._sub_portfolios.get(ref)
            if sub is not None:
                ret = ret + w * sub.return_(state, t, dt)
        return ret


# ─── Currency Hedge Portfolio ───


class CurrencyHedgePortfolio:
    """FX-hedged wrapper around another portfolio.

    Applies a hedge ratio to the FX return component.

    Args:
        underlying: The portfolio being hedged.
        hedge_ratio: Fraction of FX exposure hedged (0 = unhedged, 1 = fully hedged).
        fx_label: Key in state for FX return (e.g. ``"fx_usd_return"``).
    """

    def __init__(
        self,
        underlying: PortfolioProtocol,
        hedge_ratio: float,
        fx_label: str,
    ) -> None:
        self._underlying = underlying
        self._hedge_ratio = hedge_ratio
        self._fx_label = fx_label

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Underlying value (FX adjustment applied in return calculation).

        Args:
            state: Simulation state.
            t: Current time.

        Returns:
            Underlying portfolio value.
        """
        return self._underlying.value(state, t)

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Hedged return = underlying return + (1 - hedge_ratio) × fx_return.

        Args:
            state: Simulation state.
            t: Period start.
            dt: Period length.

        Returns:
            FX-hedged portfolio return.
        """
        base_ret = self._underlying.return_(state, t, dt)
        fx_ret = state.get(
            self._fx_label, jnp.asarray(0.0, dtype=jnp.float64)
        )
        fx_exposure = (1.0 - self._hedge_ratio) * jnp.asarray(
            fx_ret, dtype=jnp.float64
        )
        return base_ret + fx_exposure


# ─── Portfolio of Portfolios ───


class PortfolioOfPortfolios:
    """Recursive composition of portfolios with weights.

    Combines multiple sub-portfolios into a single portfolio with
    explicit weights.

    Args:
        portfolios: List of ``(weight, portfolio)`` pairs.
    """

    def __init__(self, portfolios: list[tuple[float, PortfolioProtocol]]) -> None:
        self._portfolios = portfolios

    def value(self, state: dict[str, Array], t: float) -> Array:
        """Weighted sum of sub-portfolio values.

        Args:
            state: Simulation state.
            t: Current time.

        Returns:
            Composite portfolio value.
        """
        val = jnp.asarray(0.0, dtype=jnp.float64)
        for w, port in self._portfolios:
            val = val + w * port.value(state, t)
        return val

    def return_(self, state: dict[str, Array], t: float, dt: float) -> Array:
        """Weighted sum of sub-portfolio returns.

        Args:
            state: Simulation state.
            t: Period start.
            dt: Period length.

        Returns:
            Composite portfolio return.
        """
        ret = jnp.asarray(0.0, dtype=jnp.float64)
        for w, port in self._portfolios:
            ret = ret + w * port.return_(state, t, dt)
        return ret
