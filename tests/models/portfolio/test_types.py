"""Tests for portfolio types — protocol conformance, value, and return."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.portfolio.holdings import (
    BondHolding,
    EquityHolding,
    ForwardHolding,
)
from hyesg.models.portfolio.rebalancer import AllocationRebalancer
from hyesg.models.portfolio.types import (
    BondPortfolio,
    CurrencyHedgePortfolio,
    DerivativePortfolio,
    EquityPortfolio,
    FundPortfolio,
    PortfolioOfPortfolios,
    PortfolioProtocol,
)

jax.config.update("jax_enable_x64", True)


# ─── Protocol Conformance ───


class TestProtocolConformance:
    """All portfolio types implement PortfolioProtocol."""

    def test_equity_portfolio_is_protocol(self) -> None:
        """EquityPortfolio satisfies PortfolioProtocol."""
        holdings = [EquityHolding("eq1", 1.0)]
        port = EquityPortfolio(holdings)
        assert isinstance(port, PortfolioProtocol)

    def test_bond_portfolio_is_protocol(self) -> None:
        """BondPortfolio satisfies PortfolioProtocol."""
        holdings = [BondHolding("bd1", 1.0, 100.0, 0.05, 5.0)]
        port = BondPortfolio(holdings, AllocationRebalancer())
        assert isinstance(port, PortfolioProtocol)

    def test_derivative_portfolio_is_protocol(self) -> None:
        """DerivativePortfolio satisfies PortfolioProtocol."""
        port = DerivativePortfolio()
        assert isinstance(port, PortfolioProtocol)

    def test_fund_portfolio_is_protocol(self) -> None:
        """FundPortfolio satisfies PortfolioProtocol."""
        port = FundPortfolio([], {})
        assert isinstance(port, PortfolioProtocol)

    def test_currency_hedge_is_protocol(self) -> None:
        """CurrencyHedgePortfolio satisfies PortfolioProtocol."""
        inner = EquityPortfolio([EquityHolding("eq", 1.0)])
        port = CurrencyHedgePortfolio(inner, 0.5, "fx_usd_return")
        assert isinstance(port, PortfolioProtocol)

    def test_portfolio_of_portfolios_is_protocol(self) -> None:
        """PortfolioOfPortfolios satisfies PortfolioProtocol."""
        inner = EquityPortfolio([EquityHolding("eq", 1.0)])
        port = PortfolioOfPortfolios([(1.0, inner)])
        assert isinstance(port, PortfolioProtocol)


# ─── Equity Portfolio ───


class TestEquityPortfolio:
    """Tests for EquityPortfolio."""

    def test_single_asset_value(self) -> None:
        """Single equity at double its initial price → value = 2.0."""
        holdings = [EquityHolding("eq1", weight=1.0, initial_price=50.0)]
        port = EquityPortfolio(holdings)
        state = {"eq1": jnp.asarray(100.0)}
        val = port.value(state, t=0.0)
        assert float(val) == pytest.approx(2.0, abs=1e-10)

    def test_weighted_return(self) -> None:
        """Weighted return from two equities."""
        holdings = [
            EquityHolding("eq1", weight=0.6),
            EquityHolding("eq2", weight=0.4),
        ]
        port = EquityPortfolio(holdings)
        state = {
            "eq1_return": jnp.asarray(0.10),
            "eq2_return": jnp.asarray(0.05),
        }
        ret = port.return_(state, t=0.0, dt=1.0)
        expected = 0.6 * 0.10 + 0.4 * 0.05
        assert float(ret) == pytest.approx(expected, abs=1e-10)

    def test_normalised_weights(self) -> None:
        """Weights are normalised even if they don't sum to 1."""
        holdings = [
            EquityHolding("eq1", weight=2.0),
            EquityHolding("eq2", weight=3.0),
        ]
        port = EquityPortfolio(holdings)
        assert port.weights["eq1"] == pytest.approx(0.4, abs=1e-10)
        assert port.weights["eq2"] == pytest.approx(0.6, abs=1e-10)

    def test_missing_return_defaults_zero(self) -> None:
        """Missing return key defaults to 0.0."""
        holdings = [EquityHolding("eq1", weight=1.0)]
        port = EquityPortfolio(holdings)
        state: dict[str, jax.Array] = {}
        ret = port.return_(state, t=0.0, dt=1.0)
        assert float(ret) == pytest.approx(0.0, abs=1e-10)


# ─── Bond Portfolio ───


class TestBondPortfolio:
    """Tests for BondPortfolio."""

    def test_value_from_state(self) -> None:
        """Bond portfolio value is weighted sum of bond prices."""
        holdings = [
            BondHolding("bd1", 0.6, 100.0, 0.05, 5.0),
            BondHolding("bd2", 0.4, 100.0, 0.03, 10.0),
        ]
        port = BondPortfolio(holdings, AllocationRebalancer())
        state = {
            "bd1": jnp.asarray(98.0),
            "bd2": jnp.asarray(95.0),
        }
        val = port.value(state, t=0.0)
        expected = 0.6 * 98.0 + 0.4 * 95.0
        assert float(val) == pytest.approx(expected, abs=1e-10)

    def test_return_from_state(self) -> None:
        """Bond portfolio return is weighted sum of bond returns."""
        holdings = [BondHolding("bd1", 1.0, 100.0, 0.05, 5.0)]
        port = BondPortfolio(holdings, AllocationRebalancer())
        state = {"bd1_return": jnp.asarray(0.02)}
        ret = port.return_(state, t=0.0, dt=0.5)
        assert float(ret) == pytest.approx(0.02, abs=1e-10)


# ─── Derivative Portfolio ───


class TestDerivativePortfolio:
    """Tests for DerivativePortfolio."""

    def test_empty_portfolio_value_zero(self) -> None:
        """Empty derivative portfolio has zero value."""
        port = DerivativePortfolio()
        val = port.value({}, t=0.0)
        assert float(val) == pytest.approx(0.0, abs=1e-10)

    def test_forward_value(self) -> None:
        """Forward holding picks up value from state."""
        fwd = ForwardHolding("oil", weight=1.0, delivery_date=1.0)
        port = DerivativePortfolio(forwards=[fwd])
        state = {"oil_fwd": jnp.asarray(50.0)}
        val = port.value(state, t=0.0)
        assert float(val) == pytest.approx(50.0, abs=1e-10)


# ─── Currency Hedge Portfolio ───


class TestCurrencyHedgePortfolio:
    """Tests for CurrencyHedgePortfolio."""

    def test_fully_hedged(self) -> None:
        """Fully hedged → FX component is zero."""
        inner = EquityPortfolio([EquityHolding("eq1", 1.0)])
        port = CurrencyHedgePortfolio(inner, hedge_ratio=1.0, fx_label="fx_ret")
        state = {
            "eq1_return": jnp.asarray(0.10),
            "fx_ret": jnp.asarray(0.05),
        }
        ret = port.return_(state, t=0.0, dt=1.0)
        # Fully hedged: no FX exposure
        assert float(ret) == pytest.approx(0.10, abs=1e-10)

    def test_unhedged(self) -> None:
        """Unhedged → full FX component added."""
        inner = EquityPortfolio([EquityHolding("eq1", 1.0)])
        port = CurrencyHedgePortfolio(inner, hedge_ratio=0.0, fx_label="fx_ret")
        state = {
            "eq1_return": jnp.asarray(0.10),
            "fx_ret": jnp.asarray(0.05),
        }
        ret = port.return_(state, t=0.0, dt=1.0)
        assert float(ret) == pytest.approx(0.15, abs=1e-10)

    def test_partial_hedge(self) -> None:
        """50% hedge → half FX exposure."""
        inner = EquityPortfolio([EquityHolding("eq1", 1.0)])
        port = CurrencyHedgePortfolio(inner, hedge_ratio=0.5, fx_label="fx_ret")
        state = {
            "eq1_return": jnp.asarray(0.10),
            "fx_ret": jnp.asarray(0.10),
        }
        ret = port.return_(state, t=0.0, dt=1.0)
        # 0.10 + 0.5 * 0.10 = 0.15
        assert float(ret) == pytest.approx(0.15, abs=1e-10)


# ─── Portfolio of Portfolios ───


class TestPortfolioOfPortfolios:
    """Tests for PortfolioOfPortfolios."""

    def test_recursive_value(self) -> None:
        """Composite value is weighted sum of sub-portfolio values."""
        eq = EquityPortfolio([EquityHolding("eq1", 1.0, initial_price=50.0)])
        bd = BondPortfolio(
            [BondHolding("bd1", 1.0, 100.0, 0.05, 5.0)],
            AllocationRebalancer(),
        )
        composite = PortfolioOfPortfolios([(0.6, eq), (0.4, bd)])
        state = {
            "eq1": jnp.asarray(100.0),  # 2x initial
            "bd1": jnp.asarray(98.0),
        }
        val = composite.value(state, t=0.0)
        expected = 0.6 * 2.0 + 0.4 * 98.0
        assert float(val) == pytest.approx(expected, abs=1e-10)

    def test_recursive_return(self) -> None:
        """Composite return is weighted sum of sub-portfolio returns."""
        eq = EquityPortfolio([EquityHolding("eq1", 1.0)])
        bd = BondPortfolio(
            [BondHolding("bd1", 1.0, 100.0, 0.05, 5.0)],
            AllocationRebalancer(),
        )
        composite = PortfolioOfPortfolios([(0.7, eq), (0.3, bd)])
        state = {
            "eq1_return": jnp.asarray(0.08),
            "bd1_return": jnp.asarray(0.02),
        }
        ret = composite.return_(state, t=0.0, dt=1.0)
        expected = 0.7 * 0.08 + 0.3 * 0.02
        assert float(ret) == pytest.approx(expected, abs=1e-10)

    def test_nested_composition(self) -> None:
        """PortfolioOfPortfolios can contain another PortfolioOfPortfolios."""
        eq = EquityPortfolio([EquityHolding("eq1", 1.0)])
        inner = PortfolioOfPortfolios([(1.0, eq)])
        outer = PortfolioOfPortfolios([(0.5, inner), (0.5, eq)])
        state = {"eq1_return": jnp.asarray(0.10)}
        ret = outer.return_(state, t=0.0, dt=1.0)
        assert float(ret) == pytest.approx(0.10, abs=1e-10)


# ─── Fund Portfolio ───


class TestFundPortfolio:
    """Tests for FundPortfolio."""

    def test_fund_delegates(self) -> None:
        """FundPortfolio delegates to sub-portfolios by ref."""
        eq = EquityPortfolio([EquityHolding("eq1", 1.0)])
        port = FundPortfolio(
            fund_refs=[("fund_a", 1.0)],
            sub_portfolios={"fund_a": eq},
        )
        state = {"eq1_return": jnp.asarray(0.05)}
        ret = port.return_(state, t=0.0, dt=1.0)
        assert float(ret) == pytest.approx(0.05, abs=1e-10)

    def test_missing_fund_ignored(self) -> None:
        """Missing fund ref contributes zero."""
        port = FundPortfolio(
            fund_refs=[("missing_fund", 1.0)],
            sub_portfolios={},
        )
        ret = port.return_({}, t=0.0, dt=1.0)
        assert float(ret) == pytest.approx(0.0, abs=1e-10)
