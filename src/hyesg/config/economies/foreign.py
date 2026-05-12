"""Foreign economy configurations (USD, EUR, JPY, EM, APAC).

Each foreign economy has a CIR2++ nominal rate model, an FX
exchange-rate model with stochastic volatility (CIR) and jumps,
and equity models as defined in C# Calibration.cs (lines 564–1210).

- USD has 11 equities (benchmark + 7 factors + Commodities + 2 infra + REITs)
- EUR has exactly 1 equity ("EU Equity")
- JPY has exactly 1 equity ("JP Equity")
- EM has 1 equity, proxied from USD nominal parameters
- APAC has 2 equities, proxied from USD nominal parameters
"""

from __future__ import annotations

from hyesg.config.economy import Economy, EconomyModelConfig


# ---------------------------------------------------------------------------
# FX stochastic vol + jump parameters per economy
# Extracted from C# Calibration.cs — these are contractual constants.
# ---------------------------------------------------------------------------

_USD_FX_PARAMS: dict[str, float] = {
    "jump_lambda": 7.466606,
    "jump_mu": 0.003,
    "jump_sigma": 0.0175,
    "vol_alpha": 1.996773,
    "vol_mu": 0.0837626 * 0.9,
    "vol_sigma": 0.1743646,
}

_EUR_FX_PARAMS: dict[str, float] = {
    "jump_lambda": 13.99381,
    "jump_mu": 0.001987,
    "jump_sigma": 0.0125,
    "vol_alpha": 1.498045,
    "vol_mu": 0.0733476 * 0.8,
    "vol_sigma": 0.1646164,
}

_JPY_FX_PARAMS: dict[str, float] = {
    "jump_lambda": 10.44924,
    "jump_mu": 0.0073822,
    "jump_sigma": 0.025,
    "vol_alpha": 3.349906,
    "vol_mu": 0.1086202 * 0.9,
    "vol_sigma": 0.3444948,
}

_EM_FX_PARAMS: dict[str, float] = {
    "jump_lambda": 14.66016,
    "jump_mu": -0.0016577,
    "jump_sigma": 0.02,
    "vol_alpha": 4.090783,
    "vol_mu": 0.0815481 * 1.2,
    "vol_sigma": 0.25,
}

_APAC_FX_PARAMS: dict[str, float] = {
    "jump_lambda": 17.3261,
    "jump_mu": -0.0024895,
    "jump_sigma": 0.0125,
    "vol_alpha": 2.162513,
    "vol_mu": 0.0723349 * 1.1,
    "vol_sigma": 0.2,
}


# ---------------------------------------------------------------------------
# USD — 11 equities (C# Calibration.cs lines 650–818)
# ---------------------------------------------------------------------------

_USD_EQUITY_NAMES: list[tuple[str, str]] = [
    # Benchmark
    ("usd_us_equity", "US Equity"),
    # Factor equities
    ("usd_us_factor_equity_size", "US FactorEquity Size"),
    ("usd_us_factor_equity_value", "US FactorEquity Value"),
    ("usd_us_factor_equity_income", "US FactorEquity Income"),
    ("usd_us_factor_equity_momentum", "US FactorEquity Momentum"),
    ("usd_us_factor_equity_quality", "US FactorEquity Quality"),
    ("usd_us_factor_equity_low_volatility", "US FactorEquity LowVolatility"),
    # Alternatives
    ("usd_commodities", "Commodities"),
    ("usd_global_listed_infra", "Global Developed Listed Infrastructure Equity"),
    ("usd_global_unlisted_infra", "Global Developed Unlisted Infrastructure Equity"),
    ("usd_global_reits", "Global REITs"),
]


def build_usd_economy() -> Economy:
    """Build USD foreign economy matching C# Calibration.cs lines 582–648.

    Returns:
        Economy with US nominal rates, FX (stochastic vol + jumps),
        11 equity models, and credit.
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
            params=_USD_FX_PARAMS,
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
# EUR — 1 equity (C# Calibration.cs lines 870–921)
# ---------------------------------------------------------------------------

_EUR_EQUITY_NAMES: list[tuple[str, str]] = [
    ("eur_eu_equity", "EU Equity"),
]


def build_eur_economy() -> Economy:
    """Build EUR foreign economy matching C# Calibration.cs lines 820–923.

    Returns:
        Economy with EUR nominal rates, FX (stochastic vol + jumps),
        1 equity model ("EU Equity"), and credit.
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
            params=_EUR_FX_PARAMS,
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
# JPY — 1 equity (C# Calibration.cs lines 975–1026)
# ---------------------------------------------------------------------------

_JPY_EQUITY_NAMES: list[tuple[str, str]] = [
    ("jpy_jp_equity", "JP Equity"),
]


def build_jpy_economy() -> Economy:
    """Build JPY foreign economy matching C# Calibration.cs lines 925–1028.

    Returns:
        Economy with JPY nominal rates, FX (stochastic vol + jumps),
        and 1 equity model ("JP Equity").
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
            params=_JPY_FX_PARAMS,
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
# EM — 1 equity, proxied from USD nominal (C# Calibration.cs lines 1030–1110)
# ---------------------------------------------------------------------------

_EM_EQUITY_NAMES: list[tuple[str, str]] = [
    ("em_equity", "EM Equity"),
]


def build_em_economy() -> Economy:
    """Build Emerging Markets foreign economy.

    C# proxies EM nominal rate parameters from USD (lines 1034–1036).
    The ``nominal_proxy`` param records this relationship.

    Returns:
        Economy with EM nominal rates (proxied from USD), FX, and 1 equity.
    """
    return Economy(
        name="EM",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="em_nominal",
            params={"nominal_proxy": "USD"},
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="em_fx",
            params=_EM_FX_PARAMS,
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
# APAC — 2 equities, proxied from USD nominal (C# Calibration.cs lines 1112–1210)
# ---------------------------------------------------------------------------

_APAC_EQUITY_NAMES: list[tuple[str, str]] = [
    ("apac_equity", "APAC Equity"),
    ("apac_developed_equity", "APAC Developed Equity"),
]


def build_apac_economy() -> Economy:
    """Build Asia-Pacific foreign economy.

    C# proxies APAC nominal rate parameters from USD (lines 1116–1118).
    The ``nominal_proxy`` param records this relationship.

    Returns:
        Economy with APAC nominal rates (proxied from USD), FX,
        and 2 equity models.
    """
    return Economy(
        name="APAC",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp",
            label="apac_nominal",
            params={"nominal_proxy": "USD"},
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm",
            label="apac_fx",
            params=_APAC_FX_PARAMS,
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
