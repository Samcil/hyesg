"""Tests for bond portfolio configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hyesg.core.enums import CreditClass, Liquidity
from hyesg.models.bond_portfolios.config import (
    BenchmarkPortfolioConfig,
    BondHoldingConfig,
    BondPortfolioConfig,
    BondType,
    MaturityType,
    RebalancingConfig,
)


class TestBondHoldingConfig:
    """BondHoldingConfig validation tests."""

    def test_government_bond_defaults(self) -> None:
        """Government bond with minimal args uses correct defaults."""
        h = BondHoldingConfig(maturity=10.0, weight=0.5)
        assert h.maturity == 10.0
        assert h.weight == 0.5
        assert h.bond_type == BondType.GOVERNMENT
        assert h.maturity_type == MaturityType.FIXED
        assert h.coupon_rate == 0.0
        assert h.coupon_frequency == 0
        assert h.is_at_par is False
        assert h.credit_class is None
        assert h.liquidity == Liquidity.HIGH
        assert h.economy == "GBP"

    def test_corporate_requires_credit_class(self) -> None:
        """Corporate bonds must specify a credit class."""
        with pytest.raises(ValidationError, match="credit_class"):
            BondHoldingConfig(
                maturity=5.0,
                weight=0.3,
                bond_type=BondType.CORPORATE,
                credit_class=None,
            )

    def test_corporate_with_credit_class(self) -> None:
        """Corporate bond with credit class is valid."""
        h = BondHoldingConfig(
            maturity=5.0,
            weight=0.3,
            bond_type=BondType.CORPORATE,
            credit_class=CreditClass.A,
            liquidity=Liquidity.MEDIUM,
        )
        assert h.credit_class == CreditClass.A
        assert h.liquidity == Liquidity.MEDIUM

    def test_frozen_immutability(self) -> None:
        """Config is frozen (immutable)."""
        h = BondHoldingConfig(maturity=10.0, weight=0.5)
        with pytest.raises(ValidationError):
            h.maturity = 20.0  # type: ignore[misc]

    def test_index_linked_bond(self) -> None:
        """Index-linked bond config is valid."""
        h = BondHoldingConfig(
            maturity=15.0,
            weight=1.0,
            bond_type=BondType.INDEX_LINKED,
            economy="GBP",
        )
        assert h.bond_type == BondType.INDEX_LINKED

    def test_swap_curve_bond(self) -> None:
        """Swap curve bond config is valid."""
        h = BondHoldingConfig(
            maturity=10.0,
            weight=1.0,
            bond_type=BondType.SWAP_CURVE,
        )
        assert h.bond_type == BondType.SWAP_CURVE

    def test_rolling_maturity(self) -> None:
        """Rolling maturity type is accepted."""
        h = BondHoldingConfig(
            maturity=1.0,
            weight=1.0,
            maturity_type=MaturityType.ROLLING,
        )
        assert h.maturity_type == MaturityType.ROLLING


class TestRebalancingConfig:
    """RebalancingConfig validation tests."""

    def test_defaults(self) -> None:
        """Default rebalancing config values."""
        r = RebalancingConfig()
        assert r.strategy == "maturity_and_allocation"
        assert r.frequency == 12
        assert r.rebalance_to_initial_maturity is True

    def test_custom_strategy(self) -> None:
        """Custom rebalancing strategy."""
        r = RebalancingConfig(strategy="buy_and_hold", frequency=6)
        assert r.strategy == "buy_and_hold"
        assert r.frequency == 6


class TestBondPortfolioConfig:
    """BondPortfolioConfig validation tests."""

    def test_minimal_portfolio(self) -> None:
        """Portfolio with a single holding."""
        h = BondHoldingConfig(maturity=10.0, weight=1.0)
        p = BondPortfolioConfig(name="test", holdings=[h])
        assert p.name == "test"
        assert len(p.holdings) == 1
        assert p.n_issues_per_tranche == 1

    def test_negative_weight_rejected(self) -> None:
        """Negative holding weights are rejected."""
        h = BondHoldingConfig(maturity=10.0, weight=-0.5)
        with pytest.raises(ValidationError, match="weight"):
            BondPortfolioConfig(name="bad", holdings=[h])

    def test_multi_holding_portfolio(self) -> None:
        """Portfolio with multiple holdings."""
        holdings = [
            BondHoldingConfig(maturity=5.0, weight=0.3),
            BondHoldingConfig(maturity=10.0, weight=0.7),
        ]
        p = BondPortfolioConfig(name="multi", holdings=holdings)
        assert len(p.holdings) == 2

    def test_corporate_portfolio_with_issues(self) -> None:
        """Corporate portfolio with multiple issues per tranche."""
        h = BondHoldingConfig(
            maturity=9.0,
            weight=1.0,
            bond_type=BondType.CORPORATE,
            credit_class=CreditClass.A,
        )
        p = BondPortfolioConfig(
            name="corp_test",
            holdings=[h],
            n_issues_per_tranche=3,
        )
        assert p.n_issues_per_tranche == 3


class TestBenchmarkPortfolioConfig:
    """BenchmarkPortfolioConfig validation tests."""

    def test_benchmark_creation(self) -> None:
        """Benchmark config wraps a portfolio config."""
        h = BondHoldingConfig(maturity=10.0, weight=1.0)
        p = BondPortfolioConfig(name="inner", holdings=[h])
        b = BenchmarkPortfolioConfig(
            name="Test Benchmark",
            description="A test",
            benchmark_code="T001",
            portfolio=p,
        )
        assert b.name == "Test Benchmark"
        assert b.benchmark_code == "T001"
        assert b.portfolio.name == "inner"
