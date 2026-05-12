"""Multi-currency bond issuer support.

Tracks bond positions across multiple currencies, enabling
FX-adjusted credit portfolio construction.
"""

from __future__ import annotations

from typing import NamedTuple


class BondIssuerCurrency(NamedTuple):
    """Currency designation for a bond issuer position.

    Attributes:
        currency: ISO currency code (e.g. "GBP", "USD", "EUR").
        nominal_curve_name: Name of the nominal yield curve model.
        credit_pool_name: Name of the credit pool this currency maps to.
    """

    currency: str
    nominal_curve_name: str
    credit_pool_name: str


class MultiCurrencyIssuer:
    """Bond issuer with positions in multiple currencies.

    Wraps multiple ``BondIssuerCurrency`` entries to support cross-currency
    credit portfolios.  GBP is the base currency; USD and EUR are foreign.

    Args:
        currencies: List of currency designations for this issuer.
        name: Identifier for this multi-currency position.
    """

    def __init__(
        self,
        currencies: list[BondIssuerCurrency],
        name: str = "multi_currency_issuer",
    ) -> None:
        self._currencies = list(currencies)
        self._name = name

    @property
    def name(self) -> str:
        """Identifier for this multi-currency issuer."""
        return self._name

    @property
    def currencies(self) -> list[BondIssuerCurrency]:
        """List of currency designations."""
        return list(self._currencies)

    @property
    def currency_codes(self) -> list[str]:
        """ISO currency codes for all positions."""
        return [c.currency for c in self._currencies]

    def get_currency(self, code: str) -> BondIssuerCurrency | None:
        """Look up a currency designation by ISO code.

        Args:
            code: ISO currency code (e.g. "USD").

        Returns:
            Matching ``BondIssuerCurrency`` or ``None``.
        """
        for c in self._currencies:
            if c.currency == code:
                return c
        return None

    @property
    def n_currencies(self) -> int:
        """Number of currencies in this position."""
        return len(self._currencies)
