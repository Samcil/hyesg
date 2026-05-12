"""Financial models for the hyesg ESG engine."""

from __future__ import annotations

from hyesg.models.equity.equity import Equity
from hyesg.models.exchange_rates.fx import FXRate
from hyesg.models.inflation.inflation import Inflation
from hyesg.models.jumps.jump_models import (
    ConstantIntensityJumpModel,
    StochasticIntensityJumpModel,
    ZeroJumpModel,
)
from hyesg.models.short_rates.cir import CIR
from hyesg.models.short_rates.cir2pp import CIR2PlusPlus
from hyesg.models.short_rates.cirpp import CIRPlusPlus
from hyesg.models.short_rates.deterministic import Deterministic
from hyesg.models.short_rates.g1pp import G1PP
from hyesg.models.credit.credit import Credit
from hyesg.models.short_rates.g2pp import G2PP
from hyesg.models.short_rates.vasicek import Vasicek
from hyesg.models.volatility.cir_vol import CIRVolatility
from hyesg.models.currencies import (
    BaseNominal,
    CurrencyAnalogy,
    DividendYield,
    EquityAndDividendYield,
    ExchangeRateAnalogyChain,
    FactorWedge,
    NominalAndExchangeRate,
    RealRate,
    RealRateAndInflation,
    StandardNominal,
    WedgeCurrencyAndInflationWedge,
)

__all__ = [
    "BaseNominal",
    "CIR",
    "CIR2PlusPlus",
    "CIRPlusPlus",
    "CIRVolatility",
    "ConstantIntensityJumpModel",
    "Credit",
    "CurrencyAnalogy",
    "Deterministic",
    "DividendYield",
    "Equity",
    "EquityAndDividendYield",
    "ExchangeRateAnalogyChain",
    "FactorWedge",
    "FXRate",
    "G1PP",
    "G2PP",
    "Inflation",
    "NominalAndExchangeRate",
    "RealRate",
    "RealRateAndInflation",
    "StandardNominal",
    "StochasticIntensityJumpModel",
    "Vasicek",
    "WedgeCurrencyAndInflationWedge",
    "ZeroJumpModel",
]
