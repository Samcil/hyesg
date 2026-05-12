"""Hard-coded calibration constants from the C# ESG.

These are the market prices of risk, aggregate equity parameters,
and salary wedge parameters that are embedded as compile-time
constants in the C# ``Calibration`` class rather than read from
Excel named ranges.

Source: ``SimulationSetups/Major Calibration/Calibration/Calibration.cs``
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# ── Aggregate Equity Parameters ──────────────────────────────────


class AggregateEquityParams(BaseModel):
    """Aggregate equity SVJD calibration parameters.

    These parameterise the single "aggregate equity" stochastic
    volatility jump-diffusion process from which all individual
    equity models are derived.

    Attributes:
        alpha: Mean-reversion speed of volatility.
        mu: Long-run mean variance.
        sigma: Vol-of-vol.
        jump_lambda: Jump arrival intensity.
        jump_mu: Mean log-jump size.
        jump_sigma: Jump size volatility.
        market_price_of_risk: Equity MPR for risk-neutral → real-world.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = 4.96766
    mu: float = 0.13944
    sigma: float = 0.48008
    jump_lambda: float = 14.17204
    jump_mu: float = -0.00475
    jump_sigma: float = 0.025
    market_price_of_risk: float = 0.36508


AGGREGATE_EQUITY = AggregateEquityParams()
"""Default aggregate equity parameters matching C# ESG."""


# ── Property Market Prices of Risk ───────────────────────────────


class PropertyMarketPricesOfRisk(BaseModel):
    """Market prices of risk for property and infrastructure assets.

    Each property type has its own MPR constant that determines
    the expected excess return in the real-world measure.

    Attributes:
        commercial: UK commercial property MPR.
        private_rented_sector: PRS (residential) property MPR.
        long_lease: Long-lease property MPR.
        social: Social housing MPR.
        reits: Global REITs MPR.
        listed_infra: Listed infrastructure MPR.
        unlisted_infra: Unlisted infrastructure MPR.
    """

    model_config = ConfigDict(frozen=True)

    commercial: float = 0.358
    private_rented_sector: float = 0.354
    long_lease: float = 0.50
    social: float = 0.485
    reits: float = 0.3373
    listed_infra: float = 0.35
    unlisted_infra: float = 0.57


PROPERTY_MPR = PropertyMarketPricesOfRisk()
"""Default property MPR constants matching C# ESG."""


# ── Alternative Asset Market Prices of Risk ──────────────────────


class AlternativesMarketPricesOfRisk(BaseModel):
    """Market prices of risk for alternative asset classes.

    Attributes:
        private_equity: Private equity MPR (targets ~4% excess over
            all-world equities).
        commodities: Commodities MPR (low to reflect low strategic
            investment value).
    """

    model_config = ConfigDict(frozen=True)

    private_equity: float = 0.501
    commodities: float = 0.152


ALTERNATIVES_MPR = AlternativesMarketPricesOfRisk()
"""Default alternatives MPR constants matching C# ESG."""


# ── Salary Wedge Constants ───────────────────────────────────────


class SalaryWedgeConstants(BaseModel):
    """Constants used in salary wedge calculation.

    The salary wedge models real salary growth as a spread over
    the real interest rate, blending from a short-term view to
    a long-term equilibrium view.

    Attributes:
        short_term_real_salary_growth: Short-term excess of real
            salary growth over real rates (negative = salary lag).
        long_term_real_salary_growth: Long-term equilibrium excess.
        sigma: Salary growth volatility.
        initial_salary_index: Starting index level.
    """

    model_config = ConfigDict(frozen=True)

    short_term_real_salary_growth: float = -0.01
    long_term_real_salary_growth: float = 0.01
    sigma: float = 0.0125
    initial_salary_index: float = 1.0


SALARY_WEDGE = SalaryWedgeConstants()
"""Default salary wedge constants matching C# ESG."""


# ── MPR lookup helper ────────────────────────────────────────────


def get_market_price_of_risk(asset_name: str) -> float:
    """Look up the market price of risk for a named asset.

    Searches property, alternatives, and aggregate equity constants.

    Args:
        asset_name: Asset identifier (case-insensitive). Recognised
            names include ``"commercial"``, ``"prs"``,
            ``"private_rented_sector"``, ``"long_lease"``,
            ``"social"``, ``"reits"``, ``"listed_infra"``,
            ``"unlisted_infra"``, ``"private_equity"``,
            ``"commodities"``, ``"aggregate_equity"``.

    Returns:
        The MPR constant for the asset.

    Raises:
        KeyError: If the asset name is not recognised.
    """
    key = asset_name.strip().lower().replace(" ", "_")

    # Property MPR lookup
    _property_map: dict[str, float] = {
        "commercial": PROPERTY_MPR.commercial,
        "private_rented_sector": PROPERTY_MPR.private_rented_sector,
        "prs": PROPERTY_MPR.private_rented_sector,
        "long_lease": PROPERTY_MPR.long_lease,
        "social": PROPERTY_MPR.social,
        "social_housing": PROPERTY_MPR.social,
        "reits": PROPERTY_MPR.reits,
        "global_reits": PROPERTY_MPR.reits,
        "listed_infra": PROPERTY_MPR.listed_infra,
        "listed_infrastructure": PROPERTY_MPR.listed_infra,
        "unlisted_infra": PROPERTY_MPR.unlisted_infra,
        "unlisted_infrastructure": PROPERTY_MPR.unlisted_infra,
    }

    # Alternatives MPR lookup
    _alternatives_map: dict[str, float] = {
        "private_equity": ALTERNATIVES_MPR.private_equity,
        "commodities": ALTERNATIVES_MPR.commodities,
    }

    if key in _property_map:
        return _property_map[key]
    if key in _alternatives_map:
        return _alternatives_map[key]
    if key in ("aggregate_equity", "equity"):
        return AGGREGATE_EQUITY.market_price_of_risk

    raise KeyError(
        f"Unknown asset '{asset_name}'. Recognised names: "
        f"{sorted(set(_property_map) | set(_alternatives_map) | {'aggregate_equity'})}"
    )
