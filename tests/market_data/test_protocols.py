"""Tests for MarketDataProvider protocol compliance."""

from __future__ import annotations

from hyesg.market_data.protocols import MarketDataProvider
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve


# ── Concrete test double ─────────────────────────────────────


class _StubProvider:
    """Minimal implementation satisfying MarketDataProvider."""

    def get_zero_curve(
        self, currency: str, date: str | None = None
    ) -> ParametricCurve:
        return ConstantCurve(0.04)

    def get_inflation_curve(
        self, currency: str, date: str | None = None
    ) -> ParametricCurve:
        return ConstantCurve(0.03)

    def get_credit_curve(
        self, rating: str, currency: str, date: str | None = None
    ) -> ParametricCurve:
        return ConstantCurve(0.005)

    def get_fx_spot(
        self, domestic: str, foreign: str, date: str | None = None
    ) -> float:
        return 1.27

    def get_equity_index(self, name: str, date: str | None = None) -> float:
        return 7500.0


class _IncompleteProvider:
    """Missing some methods → should NOT satisfy the protocol."""

    def get_zero_curve(
        self, currency: str, date: str | None = None
    ) -> ParametricCurve:
        return ConstantCurve(0.04)


# ── Protocol compliance tests ────────────────────────────────


class TestMarketDataProviderProtocol:
    """Verify structural sub-typing for MarketDataProvider."""

    def test_stub_is_instance(self) -> None:
        """A complete stub satisfies the runtime check."""
        assert isinstance(_StubProvider(), MarketDataProvider)

    def test_incomplete_is_not_instance(self) -> None:
        """A class missing methods does NOT satisfy the protocol."""
        assert not isinstance(_IncompleteProvider(), MarketDataProvider)

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol must be decorated with @runtime_checkable."""
        assert hasattr(MarketDataProvider, "__protocol_attrs__") or hasattr(
            MarketDataProvider, "__abstractmethods__"
        )

    def test_stub_zero_curve_returns_parametric_curve(self) -> None:
        provider = _StubProvider()
        curve = provider.get_zero_curve("GBP")
        assert isinstance(curve, ParametricCurve)

    def test_stub_inflation_curve_returns_parametric_curve(self) -> None:
        provider = _StubProvider()
        curve = provider.get_inflation_curve("GBP")
        assert isinstance(curve, ParametricCurve)

    def test_stub_credit_curve_returns_parametric_curve(self) -> None:
        provider = _StubProvider()
        curve = provider.get_credit_curve("AAA", "GBP")
        assert isinstance(curve, ParametricCurve)

    def test_stub_fx_spot_returns_float(self) -> None:
        provider = _StubProvider()
        assert isinstance(provider.get_fx_spot("GBP", "USD"), float)

    def test_stub_equity_index_returns_float(self) -> None:
        provider = _StubProvider()
        assert isinstance(provider.get_equity_index("FTSE100"), float)
