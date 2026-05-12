"""Tests for foreign economy configurations."""

from __future__ import annotations

import pytest

from hyesg.config.economies.foreign import (
    build_apac_economy,
    build_em_economy,
    build_eur_economy,
    build_jpy_economy,
    build_usd_economy,
)
from hyesg.config.economy import Economy


# ---------------------------------------------------------------------------
# Parametrised tests across all foreign economies
# ---------------------------------------------------------------------------

_ALL_BUILDERS = [
    build_usd_economy,
    build_eur_economy,
    build_jpy_economy,
    build_em_economy,
    build_apac_economy,
]


@pytest.mark.parametrize(
    "builder",
    _ALL_BUILDERS,
    ids=["USD", "EUR", "JPY", "EM", "APAC"],
)
class TestAllForeignEconomies:
    """Tests that apply to every foreign economy."""

    def test_returns_economy(self, builder) -> None:  # type: ignore[no-untyped-def]
        """Builder returns an Economy instance."""
        assert isinstance(builder(), Economy)

    def test_not_domestic(self, builder) -> None:  # type: ignore[no-untyped-def]
        """All foreign economies are non-domestic."""
        assert builder().is_domestic is False

    def test_has_nominal_rate(self, builder) -> None:  # type: ignore[no-untyped-def]
        """All foreign economies have a CIR2++ nominal rate model."""
        econ = builder()
        assert econ.nominal_rate_model.model_type == "cir2pp"

    def test_has_fx_model(self, builder) -> None:  # type: ignore[no-untyped-def]
        """All foreign economies have an FX GBM model."""
        econ = builder()
        assert econ.fx_model is not None
        assert econ.fx_model.model_type == "fx_gbm"

    def test_has_equities(self, builder) -> None:  # type: ignore[no-untyped-def]
        """All foreign economies have at least one equity model."""
        econ = builder()
        assert len(econ.equity_models) >= 1

    def test_no_real_rate(self, builder) -> None:  # type: ignore[no-untyped-def]
        """Foreign economies don't have real rate models."""
        assert builder().real_rate_model is None

    def test_no_inflation(self, builder) -> None:  # type: ignore[no-untyped-def]
        """Foreign economies don't have inflation models."""
        assert builder().inflation_model is None

    def test_no_salary(self, builder) -> None:  # type: ignore[no-untyped-def]
        """Foreign economies don't have salary models."""
        assert builder().salary_model is None


# ---------------------------------------------------------------------------
# Economy-specific tests
# ---------------------------------------------------------------------------


class TestUSDEconomy:
    """USD-specific tests matching C# Calibration.cs lines 564–818."""

    def test_name(self) -> None:
        assert build_usd_economy().name == "USD"

    def test_has_credit(self) -> None:
        """USD has a credit pool."""
        assert build_usd_economy().credit_pool is not None

    def test_has_11_equities(self) -> None:
        """USD has 11 equity models (benchmark + 6 factors + 4 alternatives)."""
        assert len(build_usd_economy().equity_models) == 11

    def test_fx_has_jump_params(self) -> None:
        """USD FX model carries stochastic vol + jump parameters."""
        fx = build_usd_economy().fx_model
        assert fx is not None
        assert fx.params is not None
        assert "jump_lambda" in fx.params
        assert "vol_alpha" in fx.params

    def test_equity_display_names(self) -> None:
        """USD equity display names match C# exactly."""
        econ = build_usd_economy()
        names = [eq.params["display_name"] for eq in econ.equity_models]
        assert names[0] == "US Equity"
        assert "Commodities" in names
        assert "Global REITs" in names


class TestEUREconomy:
    """EUR-specific tests matching C# Calibration.cs lines 820–923."""

    def test_name(self) -> None:
        assert build_eur_economy().name == "EUR"

    def test_has_credit(self) -> None:
        """EUR has a credit pool."""
        assert build_eur_economy().credit_pool is not None

    def test_has_1_equity(self) -> None:
        """EUR has exactly 1 equity model ('EU Equity')."""
        econ = build_eur_economy()
        assert len(econ.equity_models) == 1
        assert econ.equity_models[0].params["display_name"] == "EU Equity"

    def test_fx_has_jump_params(self) -> None:
        """EUR FX model carries stochastic vol + jump parameters."""
        fx = build_eur_economy().fx_model
        assert fx is not None
        assert fx.params is not None
        assert "jump_lambda" in fx.params


class TestJPYEconomy:
    """JPY-specific tests matching C# Calibration.cs lines 925–1028."""

    def test_name(self) -> None:
        assert build_jpy_economy().name == "JPY"

    def test_no_credit(self) -> None:
        """JPY has no credit pool."""
        assert build_jpy_economy().credit_pool is None

    def test_has_1_equity(self) -> None:
        """JPY has exactly 1 equity model ('JP Equity')."""
        econ = build_jpy_economy()
        assert len(econ.equity_models) == 1
        assert econ.equity_models[0].params["display_name"] == "JP Equity"


class TestEMEconomy:
    """EM-specific tests matching C# Calibration.cs lines 1030–1110."""

    def test_name(self) -> None:
        assert build_em_economy().name == "EM"

    def test_no_credit(self) -> None:
        """EM has no credit pool."""
        assert build_em_economy().credit_pool is None

    def test_has_1_equity(self) -> None:
        """EM has exactly 1 equity model ('EM Equity')."""
        econ = build_em_economy()
        assert len(econ.equity_models) == 1
        assert econ.equity_models[0].params["display_name"] == "EM Equity"

    def test_nominal_proxy_from_usd(self) -> None:
        """EM nominal model records USD proxy relationship."""
        econ = build_em_economy()
        assert econ.nominal_rate_model.params is not None
        assert econ.nominal_rate_model.params.get("nominal_proxy") == "USD"


class TestAPACEconomy:
    """APAC-specific tests matching C# Calibration.cs lines 1112–1210."""

    def test_name(self) -> None:
        assert build_apac_economy().name == "APAC"

    def test_no_credit(self) -> None:
        """APAC has no credit pool."""
        assert build_apac_economy().credit_pool is None

    def test_has_2_equities(self) -> None:
        """APAC has exactly 2 equity models."""
        econ = build_apac_economy()
        assert len(econ.equity_models) == 2

    def test_equity_display_names(self) -> None:
        """APAC equity display names match C# exactly."""
        econ = build_apac_economy()
        names = [eq.params["display_name"] for eq in econ.equity_models]
        assert names == ["APAC Equity", "APAC Developed Equity"]

    def test_nominal_proxy_from_usd(self) -> None:
        """APAC nominal model records USD proxy relationship."""
        econ = build_apac_economy()
        assert econ.nominal_rate_model.params is not None
        assert econ.nominal_rate_model.params.get("nominal_proxy") == "USD"
