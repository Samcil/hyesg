"""Tests for calibration constants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hyesg.market_data.constants import (
    AGGREGATE_EQUITY,
    ALTERNATIVES_MPR,
    PROPERTY_MPR,
    SALARY_WEDGE,
    get_market_price_of_risk,
)


class TestAggregateEquityParams:
    """Verify aggregate equity constants match C# ESG."""

    def test_alpha(self) -> None:
        assert AGGREGATE_EQUITY.alpha == pytest.approx(4.96766)

    def test_mu(self) -> None:
        assert AGGREGATE_EQUITY.mu == pytest.approx(0.13944)

    def test_sigma(self) -> None:
        assert AGGREGATE_EQUITY.sigma == pytest.approx(0.48008)

    def test_jump_lambda(self) -> None:
        assert AGGREGATE_EQUITY.jump_lambda == pytest.approx(14.17204)

    def test_jump_mu(self) -> None:
        assert AGGREGATE_EQUITY.jump_mu == pytest.approx(-0.00475)

    def test_jump_sigma(self) -> None:
        assert AGGREGATE_EQUITY.jump_sigma == pytest.approx(0.025)

    def test_market_price_of_risk(self) -> None:
        assert AGGREGATE_EQUITY.market_price_of_risk == pytest.approx(0.36508)

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            AGGREGATE_EQUITY.alpha = 1.0  # type: ignore[misc]


class TestPropertyMPR:
    """Verify property MPR constants match C# ESG."""

    def test_commercial(self) -> None:
        assert PROPERTY_MPR.commercial == pytest.approx(0.358)

    def test_prs(self) -> None:
        assert PROPERTY_MPR.private_rented_sector == pytest.approx(0.354)

    def test_long_lease(self) -> None:
        assert PROPERTY_MPR.long_lease == pytest.approx(0.50)

    def test_social(self) -> None:
        assert PROPERTY_MPR.social == pytest.approx(0.485)

    def test_reits(self) -> None:
        assert PROPERTY_MPR.reits == pytest.approx(0.3373)

    def test_listed_infra(self) -> None:
        assert PROPERTY_MPR.listed_infra == pytest.approx(0.35)

    def test_unlisted_infra(self) -> None:
        assert PROPERTY_MPR.unlisted_infra == pytest.approx(0.57)


class TestAlternativesMPR:
    """Verify alternatives MPR constants match C# ESG."""

    def test_private_equity(self) -> None:
        assert ALTERNATIVES_MPR.private_equity == pytest.approx(0.501)

    def test_commodities(self) -> None:
        assert ALTERNATIVES_MPR.commodities == pytest.approx(0.152)


class TestSalaryWedge:
    """Verify salary wedge constants match C# ESG."""

    def test_short_term(self) -> None:
        assert SALARY_WEDGE.short_term_real_salary_growth == pytest.approx(
            -0.01
        )

    def test_long_term(self) -> None:
        assert SALARY_WEDGE.long_term_real_salary_growth == pytest.approx(0.01)

    def test_sigma(self) -> None:
        assert SALARY_WEDGE.sigma == pytest.approx(0.0125)

    def test_initial_index(self) -> None:
        assert SALARY_WEDGE.initial_salary_index == pytest.approx(1.0)


class TestGetMarketPriceOfRisk:
    """Test the MPR lookup helper."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("commercial", 0.358),
            ("Commercial", 0.358),
            ("prs", 0.354),
            ("private_rented_sector", 0.354),
            ("long_lease", 0.50),
            ("social", 0.485),
            ("reits", 0.3373),
            ("listed_infra", 0.35),
            ("unlisted_infra", 0.57),
            ("private_equity", 0.501),
            ("commodities", 0.152),
            ("aggregate_equity", 0.36508),
            ("equity", 0.36508),
        ],
    )
    def test_known_assets(self, name: str, expected: float) -> None:
        assert get_market_price_of_risk(name) == pytest.approx(expected)

    def test_unknown_asset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown asset"):
            get_market_price_of_risk("nonexistent_asset")

    def test_case_insensitive(self) -> None:
        assert get_market_price_of_risk("COMMERCIAL") == pytest.approx(0.358)

    def test_space_to_underscore(self) -> None:
        assert get_market_price_of_risk("listed infra") == pytest.approx(0.35)
