"""Tests for bond grid construction."""

from __future__ import annotations

from hyesg.core.enums import CreditClass, Liquidity
from hyesg.models.bond_portfolios.config import BondType, MaturityType
from hyesg.models.bond_portfolios.grid import (
    STANDARD_COUPONS,
    STANDARD_TENORS,
    build_bond_grid,
    build_corporate_grid,
    build_government_grid,
    build_index_linked_grid,
)


class TestBuildBondGrid:
    """Tests for the generic grid builder."""

    def test_default_grid_size(self) -> None:
        """Default grid produces tenor × coupon combinations."""
        grid = build_bond_grid()
        expected = len(STANDARD_TENORS) * len(STANDARD_COUPONS)
        assert len(grid) == expected

    def test_custom_grid_size(self) -> None:
        """Custom tenors and coupons produce correct count."""
        tenors = [5.0, 10.0]
        coupons = [0.0, 0.04]
        grid = build_bond_grid(tenors=tenors, coupons=coupons)
        assert len(grid) == 4

    def test_equal_weights(self) -> None:
        """All holdings have equal weight summing to 1."""
        grid = build_bond_grid(tenors=[1.0, 5.0], coupons=[0.0, 0.02])
        total = sum(h.weight for h in grid)
        assert abs(total - 1.0) < 1e-12

    def test_zcb_has_zero_coupon_frequency(self) -> None:
        """Zero-coupon bonds have coupon_frequency = 0."""
        grid = build_bond_grid(tenors=[10.0], coupons=[0.0])
        assert len(grid) == 1
        assert grid[0].coupon_frequency == 0
        assert grid[0].is_at_par is False

    def test_coupon_bond_has_semi_annual(self) -> None:
        """Coupon bonds have frequency = 2."""
        grid = build_bond_grid(tenors=[10.0], coupons=[0.04])
        assert len(grid) == 1
        assert grid[0].coupon_frequency == 2
        assert grid[0].is_at_par is True

    def test_bond_type_propagated(self) -> None:
        """Bond type is correctly set on all holdings."""
        grid = build_bond_grid(
            tenors=[5.0],
            coupons=[0.0],
            bond_type=BondType.INDEX_LINKED,
        )
        assert grid[0].bond_type == BondType.INDEX_LINKED

    def test_maturity_type_propagated(self) -> None:
        """Maturity type is correctly set."""
        grid = build_bond_grid(
            tenors=[1.0],
            coupons=[0.0],
            maturity_type=MaturityType.ROLLING,
        )
        assert grid[0].maturity_type == MaturityType.ROLLING

    def test_economy_propagated(self) -> None:
        """Economy is correctly set."""
        grid = build_bond_grid(
            tenors=[5.0],
            coupons=[0.0],
            economy="USD",
        )
        assert grid[0].economy == "USD"

    def test_empty_tenors(self) -> None:
        """Empty tenor list produces no holdings."""
        grid = build_bond_grid(tenors=[], coupons=[0.04])
        assert len(grid) == 0

    def test_empty_coupons(self) -> None:
        """Empty coupon list produces no holdings."""
        grid = build_bond_grid(tenors=[5.0], coupons=[])
        assert len(grid) == 0


class TestGovernmentGrid:
    """Tests for the government grid builder."""

    def test_size(self) -> None:
        """Produces standard tenor × coupon grid."""
        grid = build_government_grid()
        expected = len(STANDARD_TENORS) * len(STANDARD_COUPONS)
        assert len(grid) == expected

    def test_bond_type(self) -> None:
        """All holdings are government bonds."""
        grid = build_government_grid()
        assert all(h.bond_type == BondType.GOVERNMENT for h in grid)

    def test_no_credit_class(self) -> None:
        """Government bonds have no credit class."""
        grid = build_government_grid()
        assert all(h.credit_class is None for h in grid)

    def test_economy(self) -> None:
        """Economy can be overridden."""
        grid = build_government_grid(economy="USD")
        assert all(h.economy == "USD" for h in grid)


class TestCorporateGrid:
    """Tests for the corporate grid builder."""

    def test_size(self) -> None:
        """Produces standard tenor × coupon grid."""
        grid = build_corporate_grid()
        expected = len(STANDARD_TENORS) * len(STANDARD_COUPONS)
        assert len(grid) == expected

    def test_bond_type(self) -> None:
        """All holdings are corporate bonds."""
        grid = build_corporate_grid()
        assert all(h.bond_type == BondType.CORPORATE for h in grid)

    def test_credit_class(self) -> None:
        """Credit class is set correctly."""
        grid = build_corporate_grid(credit_class=CreditClass.BBB)
        assert all(h.credit_class == CreditClass.BBB for h in grid)

    def test_liquidity(self) -> None:
        """Liquidity is set correctly."""
        grid = build_corporate_grid(liquidity=Liquidity.LOW)
        assert all(h.liquidity == Liquidity.LOW for h in grid)


class TestIndexLinkedGrid:
    """Tests for the index-linked grid builder."""

    def test_size(self) -> None:
        """Produces standard tenor × coupon grid."""
        grid = build_index_linked_grid()
        expected = len(STANDARD_TENORS) * len(STANDARD_COUPONS)
        assert len(grid) == expected

    def test_bond_type(self) -> None:
        """All holdings are index-linked."""
        grid = build_index_linked_grid()
        assert all(h.bond_type == BondType.INDEX_LINKED for h in grid)
