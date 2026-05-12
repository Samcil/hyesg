"""Tests for Economy and EconomyModelConfig dataclasses."""

from __future__ import annotations

import pytest

from hyesg.config.economy import Economy, EconomyModelConfig


# ---------------------------------------------------------------------------
# EconomyModelConfig tests
# ---------------------------------------------------------------------------


class TestEconomyModelConfig:
    """Tests for EconomyModelConfig."""

    def test_stores_model_type(self) -> None:
        """model_type is stored correctly."""
        cfg = EconomyModelConfig(model_type="cir2pp", label="gbp_nominal")
        assert cfg.model_type == "cir2pp"

    def test_stores_label(self) -> None:
        """label is stored correctly."""
        cfg = EconomyModelConfig(model_type="gbm", label="gbp_uk_eq")
        assert cfg.label == "gbp_uk_eq"

    def test_stores_params(self) -> None:
        """params dict is stored and accessible."""
        cfg = EconomyModelConfig(
            model_type="fca",
            label="gbp_inflation",
            params={"underlying": "gbp_real"},
        )
        assert cfg.params == {"underlying": "gbp_real"}

    def test_params_default_empty(self) -> None:
        """params defaults to empty dict."""
        cfg = EconomyModelConfig(model_type="cir2pp", label="test")
        assert cfg.params == {}

    def test_frozen(self) -> None:
        """EconomyModelConfig is immutable (frozen)."""
        cfg = EconomyModelConfig(model_type="cir2pp", label="test")
        with pytest.raises(Exception):
            cfg.label = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Economy tests
# ---------------------------------------------------------------------------


class TestEconomy:
    """Tests for Economy dataclass."""

    @pytest.fixture()
    def minimal_economy(self) -> Economy:
        """Economy with only a nominal rate model."""
        return Economy(
            name="TEST",
            nominal_rate_model=EconomyModelConfig(
                model_type="cir2pp", label="test_nominal"
            ),
        )

    @pytest.fixture()
    def full_economy(self) -> Economy:
        """Economy with all optional models populated."""
        return Economy(
            name="FULL",
            is_domestic=True,
            nominal_rate_model=EconomyModelConfig(
                model_type="cir2pp", label="full_nominal"
            ),
            real_rate_model=EconomyModelConfig(
                model_type="g2pp", label="full_real"
            ),
            inflation_model=EconomyModelConfig(
                model_type="fca", label="full_inflation"
            ),
            fx_model=EconomyModelConfig(
                model_type="fx_gbm", label="full_fx"
            ),
            equity_models=[
                EconomyModelConfig(model_type="gbm", label="full_eq1"),
                EconomyModelConfig(model_type="gbm", label="full_eq2"),
            ],
            credit_pool=EconomyModelConfig(
                model_type="cir_credit", label="full_credit"
            ),
            salary_model=EconomyModelConfig(
                model_type="g2pp", label="full_salary"
            ),
        )

    def test_requires_name(self) -> None:
        """Economy requires a name."""
        with pytest.raises(Exception):
            Economy(  # type: ignore[call-arg]
                nominal_rate_model=EconomyModelConfig(
                    model_type="cir2pp", label="test"
                ),
            )

    def test_requires_nominal_rate(self) -> None:
        """Economy requires a nominal_rate_model."""
        with pytest.raises(Exception):
            Economy(name="BAD")  # type: ignore[call-arg]

    def test_is_domestic_default_false(self, minimal_economy: Economy) -> None:
        """is_domestic defaults to False."""
        assert minimal_economy.is_domestic is False

    def test_is_domestic_flag(self, full_economy: Economy) -> None:
        """is_domestic can be set to True."""
        assert full_economy.is_domestic is True

    def test_all_models_minimal(self, minimal_economy: Economy) -> None:
        """all_models returns just nominal for minimal economy."""
        models = minimal_economy.all_models
        assert len(models) == 1
        assert models[0].label == "test_nominal"

    def test_all_models_full_count(self, full_economy: Economy) -> None:
        """all_models returns all 7 models for full economy."""
        # nominal + fx + real + inflation + 2 eq + credit + salary = 8
        models = full_economy.all_models
        assert len(models) == 8

    def test_all_models_dependency_order(self, full_economy: Economy) -> None:
        """all_models returns models in dependency order."""
        labels = [m.label for m in full_economy.all_models]
        expected = [
            "full_nominal",
            "full_fx",
            "full_real",
            "full_inflation",
            "full_eq1",
            "full_eq2",
            "full_credit",
            "full_salary",
        ]
        assert labels == expected

    def test_optional_fields_default_none(self, minimal_economy: Economy) -> None:
        """Optional fields default to None."""
        assert minimal_economy.real_rate_model is None
        assert minimal_economy.inflation_model is None
        assert minimal_economy.fx_model is None
        assert minimal_economy.credit_pool is None
        assert minimal_economy.salary_model is None

    def test_equity_models_default_empty(self, minimal_economy: Economy) -> None:
        """equity_models defaults to empty list."""
        assert minimal_economy.equity_models == []

    def test_frozen(self, minimal_economy: Economy) -> None:
        """Economy is immutable (frozen)."""
        with pytest.raises(Exception):
            minimal_economy.name = "changed"  # type: ignore[misc]

    def test_name_stored(self) -> None:
        """Economy name is stored correctly."""
        econ = Economy(
            name="GBP",
            nominal_rate_model=EconomyModelConfig(
                model_type="cir2pp", label="gbp_nominal"
            ),
        )
        assert econ.name == "GBP"
