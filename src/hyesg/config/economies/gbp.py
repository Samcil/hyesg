"""GBP domestic economy configuration.

Builds the GBP economy with all model types matching the C# ESS
calibration structure: CIR2++ nominal rates, G2++ real rates,
FCA-based inflation, 14 UK equity/growth-asset models, credit
pool, and salary wedge.
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


# UK equity/growth-asset labels matching C# ESS exactly.
_UK_EQUITY_NAMES: list[tuple[str, str]] = [
    ("gbp_uk_eq_benchmark", "UK Eq Benchmark"),
    ("gbp_uk_eq_size", "UK Eq Size"),
    ("gbp_uk_eq_value", "UK Eq Value"),
    ("gbp_uk_eq_low_vol", "UK Eq Low Vol"),
    ("gbp_uk_eq_quality", "UK Eq Quality"),
    ("gbp_direct_property", "Direct Property"),
    ("gbp_dput", "DPUT"),
    ("gbp_secondary_property", "Secondary Property"),
    ("gbp_long_lease", "Long Lease"),
    ("gbp_reits", "REITs"),
    ("gbp_absolute_return", "Absolute Return"),
    ("gbp_dgf3", "DGF3"),
    ("gbp_private_equity", "Private Equity"),
    ("gbp_venture_capital", "Venture Capital"),
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
