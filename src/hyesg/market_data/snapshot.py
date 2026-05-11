"""Point-in-time market data snapshot."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hyesg.math.curves.protocol import ParametricCurve

if TYPE_CHECKING:
    from hyesg.market_data.protocols import MarketDataProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Frozen point-in-time market data for simulation.

    All curves and spot values are captured once and then immutable,
    ensuring consistency throughout a simulation run.

    Attributes:
        zero_curves: Currency → nominal zero curve.
        inflation_curves: Currency → real (inflation) zero curve.
        credit_curves: Rating → currency → credit spread curve.
        fx_spots: Currency-pair string → spot rate.
        equity_indices: Index name → level.
    """

    zero_curves: dict[str, ParametricCurve] = field(default_factory=dict)
    inflation_curves: dict[str, ParametricCurve] = field(default_factory=dict)
    credit_curves: dict[str, dict[str, ParametricCurve]] = field(default_factory=dict)
    fx_spots: dict[str, float] = field(default_factory=dict)
    equity_indices: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_provider(
        cls,
        provider: MarketDataProvider,
        currencies: list[str] | None = None,
        credit_ratings: list[str] | None = None,
        credit_currencies: list[str] | None = None,
        fx_pairs: list[tuple[str, str]] | None = None,
        equity_names: list[str] | None = None,
        date: str | None = None,
    ) -> MarketDataSnapshot:
        """Build a snapshot by querying a :class:`MarketDataProvider`.

        Only the requested items are loaded. Pass ``None`` or empty
        lists for categories that are not needed.

        Args:
            provider: Market data source.
            currencies: Currencies for which to load zero and
                inflation curves.
            credit_ratings: Credit ratings to load.
            credit_currencies: Currencies for credit curves (combined
                with each rating).
            fx_pairs: ``(domestic, foreign)`` tuples for FX spots.
            equity_names: Equity index names.
            date: Valuation date passed to the provider.

        Returns:
            A frozen :class:`MarketDataSnapshot`.
        """
        currencies = currencies or []
        credit_ratings = credit_ratings or []
        credit_currencies = credit_currencies or []
        fx_pairs = fx_pairs or []
        equity_names = equity_names or []

        zero_curves: dict[str, ParametricCurve] = {}
        inflation_curves: dict[str, ParametricCurve] = {}

        for ccy in currencies:
            try:
                zero_curves[ccy] = provider.get_zero_curve(ccy, date)
            except Exception:
                logger.warning("Failed to load zero curve for %s", ccy)
            try:
                inflation_curves[ccy] = provider.get_inflation_curve(ccy, date)
            except Exception:
                logger.warning("Failed to load inflation curve for %s", ccy)

        credit_map: dict[str, dict[str, ParametricCurve]] = {}
        for rating in credit_ratings:
            rating_curves: dict[str, ParametricCurve] = {}
            for ccy in credit_currencies:
                try:
                    rating_curves[ccy] = provider.get_credit_curve(
                        rating, ccy, date
                    )
                except Exception:
                    logger.warning(
                        "Failed to load credit curve for %s/%s", rating, ccy
                    )
            if rating_curves:
                credit_map[rating] = rating_curves

        fx_spots: dict[str, float] = {}
        for dom, fgn in fx_pairs:
            pair_key = f"{dom}{fgn}"
            try:
                fx_spots[pair_key] = provider.get_fx_spot(dom, fgn, date)
            except Exception:
                logger.warning("Failed to load FX spot for %s", pair_key)

        equity_indices: dict[str, float] = {}
        for name in equity_names:
            try:
                equity_indices[name] = provider.get_equity_index(name, date)
            except Exception:
                logger.warning("Failed to load equity index %s", name)

        return cls(
            zero_curves=zero_curves,
            inflation_curves=inflation_curves,
            credit_curves=credit_map,
            fx_spots=fx_spots,
            equity_indices=equity_indices,
        )

    @property
    def currencies(self) -> list[str]:
        """List of currencies with zero curves loaded."""
        return sorted(self.zero_curves.keys())

    @property
    def ratings(self) -> list[str]:
        """List of credit ratings loaded."""
        return sorted(self.credit_curves.keys())
