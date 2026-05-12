"""Tests for GBP domestic economy configuration."""

from __future__ import annotations

from hyesg.config.economies.gbp import build_gbp_economy
from hyesg.config.economy import Economy


class TestGBPEconomy:
    """Tests for the GBP domestic economy builder."""

    def test_returns_economy(self) -> None:
        """build_gbp_economy returns an Economy instance."""
        econ = build_gbp_economy()
        assert isinstance(econ, Economy)

    def test_name_is_gbp(self) -> None:
        """Economy name is 'GBP'."""
        econ = build_gbp_economy()
        assert econ.name == "GBP"

    def test_is_domestic(self) -> None:
        """GBP is the domestic economy."""
        econ = build_gbp_economy()
        assert econ.is_domestic is True

    def test_has_nominal_rate(self) -> None:
        """GBP has a CIR2++ nominal rate model."""
        econ = build_gbp_economy()
        assert econ.nominal_rate_model.model_type == "cir2pp"
        assert econ.nominal_rate_model.label == "gbp_nominal"

    def test_has_real_rate(self) -> None:
        """GBP has a G2++ real rate model."""
        econ = build_gbp_economy()
        assert econ.real_rate_model is not None
        assert econ.real_rate_model.model_type == "g2pp"
        assert econ.real_rate_model.label == "gbp_real"

    def test_has_inflation(self) -> None:
        """GBP has an FCA inflation model."""
        econ = build_gbp_economy()
        assert econ.inflation_model is not None
        assert econ.inflation_model.model_type == "fca"
        assert econ.inflation_model.label == "gbp_inflation"

    def test_has_14_equities(self) -> None:
        """GBP has exactly 14 equity/growth-asset models."""
        econ = build_gbp_economy()
        assert len(econ.equity_models) == 14

    def test_equity_labels_unique(self) -> None:
        """All equity labels are unique."""
        econ = build_gbp_economy()
        labels = [eq.label for eq in econ.equity_models]
        assert len(labels) == len(set(labels))

    def test_equity_types_all_gbm(self) -> None:
        """All equity models are GBM type."""
        econ = build_gbp_economy()
        for eq in econ.equity_models:
            assert eq.model_type == "gbm"

    def test_has_credit(self) -> None:
        """GBP has a credit pool model."""
        econ = build_gbp_economy()
        assert econ.credit_pool is not None
        assert econ.credit_pool.model_type == "cir_credit"
        assert econ.credit_pool.label == "gbp_credit"

    def test_has_salary(self) -> None:
        """GBP has a salary model."""
        econ = build_gbp_economy()
        assert econ.salary_model is not None
        assert econ.salary_model.model_type == "g2pp"
        assert econ.salary_model.label == "gbp_salary"

    def test_no_fx_model(self) -> None:
        """GBP (domestic) has no FX model."""
        econ = build_gbp_economy()
        assert econ.fx_model is None

    def test_all_models_count(self) -> None:
        """all_models returns all 19 models.

        nominal + real + inflation + 14 eq + credit + salary = 19.
        """
        econ = build_gbp_economy()
        assert len(econ.all_models) == 19

    def test_equity_includes_benchmark(self) -> None:
        """UK Eq Benchmark is in the equity models."""
        econ = build_gbp_economy()
        labels = [eq.label for eq in econ.equity_models]
        assert "gbp_uk_eq_benchmark" in labels

    def test_equity_includes_property_types(self) -> None:
        """Property-related models are present."""
        econ = build_gbp_economy()
        labels = {eq.label for eq in econ.equity_models}
        assert "gbp_direct_property" in labels
        assert "gbp_dput" in labels
        assert "gbp_secondary_property" in labels
        assert "gbp_long_lease" in labels
        assert "gbp_reits" in labels
