"""Market data provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hyesg.math.curves.protocol import ParametricCurve


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data access.

    Implementations provide term-structure curves and spot values
    for use in ESG simulation calibration and initialisation.
    """

    def get_zero_curve(
        self,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the nominal zero-coupon yield curve.

        Args:
            currency: ISO currency code (e.g. ``"GBP"``).
            date: Optional valuation date (ISO format). If ``None``,
                use the latest available data.

        Returns:
            A curve mapping tenor (years) to continuously-compounded
            zero rate.
        """
        ...

    def get_inflation_curve(
        self,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the real (inflation-linked) yield curve.

        Args:
            currency: ISO currency code.
            date: Optional valuation date.

        Returns:
            A curve mapping tenor to real zero rate.
        """
        ...

    def get_credit_curve(
        self,
        rating: str,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the credit spread curve for a given rating.

        Args:
            rating: Credit rating label (e.g. ``"AAA"``, ``"BBB"``).
            currency: ISO currency code.
            date: Optional valuation date.

        Returns:
            A curve mapping tenor to credit spread.
        """
        ...

    def get_fx_spot(
        self,
        domestic: str,
        foreign: str,
        date: str | None = None,
    ) -> float:
        """Return the FX spot rate (units of domestic per 1 foreign).

        Args:
            domestic: Domestic currency code.
            foreign: Foreign currency code.
            date: Optional valuation date.

        Returns:
            Spot exchange rate.
        """
        ...

    def get_equity_index(
        self,
        name: str,
        date: str | None = None,
    ) -> float:
        """Return the level of an equity index.

        Args:
            name: Index identifier (e.g. ``"FTSE100"``).
            date: Optional valuation date.

        Returns:
            Index level.
        """
        ...
