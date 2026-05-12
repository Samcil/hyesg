"""GBP domestic economy configuration.

Builds the GBP economy with all model types matching the C# ESS
calibration structure (Calibration.cs lines 312–562):

- CIR2++ nominal rates, G2++ real rates, FCA-based inflation
- 14 UK equity/growth-asset models:
  8 listed equities (benchmark + 7 factors) + 6 property/alternatives
- Credit pool and salary wedge
"""

from __future__ import annotations

from hyesg.config.economy import Economy, EconomyModelConfig


def build_gbp_economy() -> Economy:
    """Build GBP domestic economy matching C# ESS structure.

    Returns:
        Economy with all UK models configured.
    """
    nominal = EconomyModelConfig(
        model_type="cir2pp",
        label="gbp_nominal",
    )

    real = EconomyModelConfig(
        model_type="g2pp",
        label="gbp_real",
    )

    inflation = EconomyModelConfig(
        model_type="fca",
        label="gbp_inflation",
        params={"underlying": "gbp_real"},
    )

    equities = _build_uk_equities()

    credit = EconomyModelConfig(
        model_type="cir_credit",
        label="gbp_credit",
    )

    salary = EconomyModelConfig(
        model_type="g2pp",
        label="gbp_salary",
        params={"underlying": "gbp_real"},
    )

    return Economy(
        name="GBP",
        is_domestic=True,
        nominal_rate_model=nominal,
        real_rate_model=real,
        inflation_model=inflation,
        equity_models=equities,
        credit_pool=credit,
        salary_model=salary,
    )


# UK equity/growth-asset labels matching C# ESS Calibration.cs exactly.
# 14 assets: 8 listed equities (benchmark + 7 factors) + 6 property/alternatives.
# Display names are contractual — they appear in output paths.
_UK_EQUITY_NAMES: list[tuple[str, str]] = [
    # Listed equities: benchmark + 7 factor equities
    ("gbp_uk_equity", "UK Equity"),
    ("gbp_uk_factor_equity_size", "UK FactorEquity Size"),
    ("gbp_uk_factor_equity_size_mid", "UK FactorEquity Size Mid"),
    ("gbp_uk_factor_equity_value", "UK FactorEquity Value"),
    ("gbp_uk_factor_equity_income", "UK FactorEquity Income"),
    ("gbp_uk_factor_equity_momentum", "UK FactorEquity Momentum"),
    ("gbp_uk_factor_equity_quality", "UK FactorEquity Quality"),
    ("gbp_uk_factor_equity_low_volatility", "UK FactorEquity LowVolatility"),
    # Property and alternatives
    ("gbp_uk_reits", "UK REITs"),
    ("gbp_private_equity_gross", "Private Equity Gross"),
    ("gbp_uk_commercial_property", "UK Commercial Property"),
    ("gbp_uk_prs_property", "UK Private Rented Sector Property"),
    ("gbp_uk_social_housing_property", "UK Social Housing Sector Property"),
    ("gbp_uk_long_lease_property", "UK Long Lease Sector Property"),
]


def _build_uk_equities() -> list[EconomyModelConfig]:
    """Build the 14 UK equity / growth-asset model configs."""
    return [
        EconomyModelConfig(
            model_type="gbm",
            label=label,
            params={"display_name": display_name},
        )
        for label, display_name in _UK_EQUITY_NAMES
    ]
