"""Canonical output name registry matching C# ESG engine conventions.

Every output key emitted by a model ``step()`` method **must** use a constant
from :class:`OutputName`.  This ensures:

* Output names match the C# ESG engine exactly (PascalCase).
* No ad-hoc string literals drift across modules.
* Renaming an output requires changing exactly one place.

Usage::

    from hyesg.outputs import OutputName

    outputs = {OutputName.SHORT_RATE: short_rate}

The class also provides helpers for dynamic output names that embed
a maturity, economy name, or portfolio name.
"""

from __future__ import annotations

from typing import Final


class OutputName:
    """Central registry of canonical ESG output names.

    Static names use ``UPPER_SNAKE`` Python constants whose **values**
    are PascalCase strings matching C# ``AddOutput`` identifiers.

    Internal step-output keys (not directly in C# Outputs.cs but used
    across Python models as inter-model interface keys) also live here
    so that *all* string literals are eliminated from model code.
    """

    # ── C# canonical output names (from Outputs.cs AddOutput calls) ──

    CASH_ACCOUNT: Final[str] = "CashAccount"
    SHORT_RATE: Final[str] = "ShortRate"
    FORWARD_RATE_CURVE_CONTINUOUS: Final[str] = "ForwardRateCurveContinuous"
    SPOT_RATE_ANNUALISED: Final[str] = "SpotRateAnnualised"
    SPOT_RATE: Final[str] = "SpotRate"
    INFLATION_INDEX: Final[str] = "InflationIndex"
    WEDGE_INDEX_BASE_CURRENCY: Final[str] = "WedgeIndexBaseCurrency"
    TOTAL_RETURN_INDEX: Final[str] = "TotalReturnIndex"
    DIVIDEND_YIELD: Final[str] = "DividendYield"
    VALUE: Final[str] = "Value"
    DURATION: Final[str] = "Duration"
    YIELD_ANNUALISED: Final[str] = "YieldAnnualised"
    EXCHANGE_RATE: Final[str] = "ExchangeRate"
    SIGMA: Final[str] = "Sigma"
    DZ: Final[str] = "dZ"
    FORWARD_FX_RATE: Final[str] = "ForwardFxRate"
    SABR_PARAMETERS: Final[str] = "SabrParameters"
    SABR_LPI_SWAP_RATE: Final[str] = "SabrLpiSwapRate"
    TARGET_SHORT_RATE_PATH: Final[str] = (
        "TargetShortRatePathForwardRateCurveContinuous"
    )
    TIME_OF_DEFAULT: Final[str] = "TimeOfDefault"

    # ── Internal step-output keys (used across Python models) ────────

    LOG_RETURN: Final[str] = "LogReturn"
    VARIANCE: Final[str] = "Variance"
    JUMP: Final[str] = "Jump"
    DRIFT_ADJUSTMENT: Final[str] = "DriftAdjustment"
    N_JUMPS: Final[str] = "NJumps"
    INTENSITY: Final[str] = "Intensity"
    SURVIVAL_PROBABILITY: Final[str] = "SurvivalProbability"
    CUM_INTENSITY: Final[str] = "CumIntensity"
    INFLATION_RATE: Final[str] = "InflationRate"
    SALARY_RATE: Final[str] = "SalaryRate"
    SALARY_INDEX: Final[str] = "SalaryIndex"
    SPOT: Final[str] = "Spot"
    UNHEDGED_RETURN: Final[str] = "UnhedgedReturn"
    HEDGE_GAIN: Final[str] = "HedgeGain"
    TRANSACTION_COST: Final[str] = "TransactionCost"
    HEDGED_RETURN: Final[str] = "HedgedReturn"

    # ── Dynamic output name builders ─────────────────────────────────

    @staticmethod
    def spot_rate(maturity: int) -> str:
        """Per-maturity spot rate: ``'{m}ySpotRate'``."""
        return f"{maturity}ySpotRate"

    @staticmethod
    def spot_rate_annualised(maturity: int) -> str:
        """Per-maturity annualised spot rate: ``'{m}ySpotRateAnnualised'``."""
        return f"{maturity}ySpotRateAnnualised"

    @staticmethod
    def exchange_rate_per_unit(economy: str) -> str:
        """Per-economy exchange rate: ``'GBPPerUnit{Name}ExchangeRate'``."""
        return f"GBPPerUnit{economy}ExchangeRate"

    @staticmethod
    def bond_portfolio_field(portfolio: str, field: str) -> str:
        """Bond portfolio output: ``'BondPortfolio({name}).{field}'``."""
        return f"BondPortfolio({portfolio}).{field}"

    @staticmethod
    def forward_tenor(tenor: int) -> str:
        """Forward FX rate for a specific tenor: ``'forward_{tenor}'``."""
        return f"forward_{tenor}"

    # ── Validation ───────────────────────────────────────────────────

    _CANONICAL: frozenset[str] | None = None

    @classmethod
    def all_canonical(cls) -> frozenset[str]:
        """Return the set of all static canonical output names."""
        if cls._CANONICAL is None:
            cls._CANONICAL = frozenset(
                v
                for k, v in vars(cls).items()
                if k.isupper() and isinstance(v, str)
            )
        return cls._CANONICAL

    @classmethod
    def validate(cls, name: str) -> bool:
        """Check whether *name* is a known canonical output name."""
        return name in cls.all_canonical()


__all__ = ["OutputName"]
