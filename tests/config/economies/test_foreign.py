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
    """USD-specific tests."""

    def test_name(self) -> None:
        assert build_usd_economy().name == "USD"

    def test_has_credit(self) -> None:
        """USD has a credit pool."""
        assert build_usd_economy().credit_pool is not None

    def test_has_5_equities(self) -> None:
        """USD has 5 US equity models (benchmark + 4 factors)."""
        assert len(build_usd_economy().equity_models) == 5


class TestEUREconomy:
    """EUR-specific tests."""

    def test_name(self) -> None:
        assert build_eur_economy().name == "EUR"

    def test_has_credit(self) -> None:
        """EUR has a credit pool."""
        assert build_eur_economy().credit_pool is not None

    def test_has_5_equities(self) -> None:
        """EUR has 5 EU equity models (benchmark + 4 factors)."""
        assert len(build_eur_economy().equity_models) == 5


class TestJPYEconomy:
    """JPY-specific tests."""

    def test_name(self) -> None:
        assert build_jpy_economy().name == "JPY"

    def test_no_credit(self) -> None:
        """JPY has no credit pool."""
        assert build_jpy_economy().credit_pool is None


class TestEMEconomy:
    """EM-specific tests."""

    def test_name(self) -> None:
        assert build_em_economy().name == "EM"

    def test_no_credit(self) -> None:
        """EM has no credit pool."""
        assert build_em_economy().credit_pool is None


class TestAPACEconomy:
    """APAC-specific tests."""

    def test_name(self) -> None:
        assert build_apac_economy().name == "APAC"

    def test_no_credit(self) -> None:
        """APAC has no credit pool."""
        assert build_apac_economy().credit_pool is None
