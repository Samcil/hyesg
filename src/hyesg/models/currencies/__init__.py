"""Foreign Currency Analogy (FCA) framework.

The FCA treats real rates, inflation, dividends, and equity factors
as 'pseudo-currencies' with exchange rates, enabling unified pricing
via zero-coupon bond prices.

Exports:
    Protocols:
        CurrencyAnalogy — protocol for all FCA types

    FCA Types:
        BaseNominal — domestic economy (Q = 1)
        StandardNominal — foreign economy (Q = FX rate)
        RealRate — real rate pseudo-currency (Q = price index)
        DividendYield — dividend yield pseudo-currency (Q = S/S₀)
        FactorWedge — equity factor pseudo-currency

    Compositions:
        NominalAndExchangeRate — nominal + FX
        RealRateAndInflation — real rate + inflation
        EquityAndDividendYield — equity + dividend
        WedgeCurrencyAndInflationWedge — RPI reform wedge

    Chain:
        ExchangeRateAnalogyChain — cross-currency chaining
"""

from __future__ import annotations

from hyesg.models.currencies.base_nominal import BaseNominal
from hyesg.models.currencies.chain import ExchangeRateAnalogyChain
from hyesg.models.currencies.compositions import (
    EquityAndDividendYield,
    NominalAndExchangeRate,
    RealRateAndInflation,
    WedgeCurrencyAndInflationWedge,
)
from hyesg.models.currencies.dividend_yield import DividendYield
from hyesg.models.currencies.factor_wedge import FactorWedge
from hyesg.models.currencies.protocols import CurrencyAnalogy
from hyesg.models.currencies.real_rate import RealRate
from hyesg.models.currencies.standard_nominal import StandardNominal

__all__ = [
    "BaseNominal",
    "CurrencyAnalogy",
    "DividendYield",
    "EquityAndDividendYield",
    "ExchangeRateAnalogyChain",
    "FactorWedge",
    "NominalAndExchangeRate",
    "RealRate",
    "RealRateAndInflation",
    "StandardNominal",
    "WedgeCurrencyAndInflationWedge",
]
