"""Foreign economy configurations (USD, EUR, JPY, EM, APAC).

Each foreign economy has at minimum a CIR2++ nominal rate model
and a GBM-based FX model. Equity and credit models are added
per the C# ESS calibration structure.
"""

from __future__ import annotations

from hyesg.config.economy import Economy, EconomyModelConfig


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------

_USD_EQUITY_NAMES: list[tuple[str, str]] = [
    ("usd_us_eq_benchmark", "US Eq Benchmark"),
    ("usd_us_eq_size", "US Eq Size"),
    ("usd_us_eq_value", "US Eq Value"),
    ("usd_us_eq_low_vol", "US Eq Low Vol"),
    ("usd_us_eq_quality", "US Eq Quality"),
]


def build_usd_economy() -> Economy:
    """Build USD foreign economy.

    Returns:
        Economy with US nominal rates, FX, equities, and credit.
    """
    return Economy(
        name="USD",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="usd_nominal",
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="usd_fx",
        ),
        equity_models=[
            EconomyModelConfig(
                model_type="gbm",
                label=label,
                params={"display_name": display_name},
            )
            for label, display_name in _USD_EQUITY_NAMES
        ],
        credit_pool=EconomyModelConfig(
            model_type="cir_credit",
            label="usd_credit",
        ),
    )


# ---------------------------------------------------------------------------
# EUR
# ---------------------------------------------------------------------------

_EUR_EQUITY_NAMES: list[tuple[str, str]] = [
    ("eur_eu_eq_benchmark", "EU Eq Benchmark"),
    ("eur_eu_eq_size", "EU Eq Size"),
    ("eur_eu_eq_value", "EU Eq Value"),
    ("eur_eu_eq_low_vol", "EU Eq Low Vol"),
    ("eur_eu_eq_quality", "EU Eq Quality"),
]


def build_eur_economy() -> Economy:
    """Build EUR foreign economy.

    Returns:
        Economy with EUR nominal rates, FX, equities, and credit.
    """
    return Economy(
        name="EUR",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="eur_nominal",
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="eur_fx",
        ),
        equity_models=[
            EconomyModelConfig(
                model_type="gbm",
                label=label,
                params={"display_name": display_name},
            )
            for label, display_name in _EUR_EQUITY_NAMES
        ],
        credit_pool=EconomyModelConfig(
            model_type="cir_credit",
            label="eur_credit",
        ),
    )


# ---------------------------------------------------------------------------
# JPY
# ---------------------------------------------------------------------------

_JPY_EQUITY_NAMES: list[tuple[str, str]] = [
    ("jpy_japan_eq_benchmark", "Japan Eq Benchmark"),
]


def build_jpy_economy() -> Economy:
    """Build JPY foreign economy.

    Returns:
        Economy with JPY nominal rates, FX, and Japan equities.
    """
    return Economy(
        name="JPY",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="jpy_nominal",
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="jpy_fx",
        ),
        equity_models=[
            EconomyModelConfig(
                model_type="gbm",
                label=label,
                params={"display_name": display_name},
            )
            for label, display_name in _JPY_EQUITY_NAMES
        ],
    )


# ---------------------------------------------------------------------------
# EM (Emerging Markets)
# ---------------------------------------------------------------------------

_EM_EQUITY_NAMES: list[tuple[str, str]] = [
    ("em_eq_benchmark", "EM Eq Benchmark"),
]


def build_em_economy() -> Economy:
    """Build Emerging Markets foreign economy.

    Returns:
        Economy with EM nominal rates, FX, and EM equities.
    """
    return Economy(
        name="EM",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="em_nominal",
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="em_fx",
        ),
        equity_models=[
            EconomyModelConfig(
                model_type="gbm",
                label=label,
                params={"display_name": display_name},
            )
            for label, display_name in _EM_EQUITY_NAMES
        ],
    )


# ---------------------------------------------------------------------------
# APAC (Asia-Pacific ex Japan)
# ---------------------------------------------------------------------------

_APAC_EQUITY_NAMES: list[tuple[str, str]] = [
    ("apac_eq_benchmark", "APAC Eq Benchmark"),
]


def build_apac_economy() -> Economy:
    """Build Asia-Pacific foreign economy.

    Returns:
        Economy with APAC nominal rates, FX, and APAC equities.
    """
    return Economy(
        name="APAC",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="apac_nominal",
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="apac_fx",
        ),
        equity_models=[
            EconomyModelConfig(
                model_type="gbm",
                label=label,
                params={"display_name": display_name},
            )
            for label, display_name in _APAC_EQUITY_NAMES
        ],
    )
