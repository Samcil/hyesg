"""Tests for MarketDataSnapshot."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyesg.market_data.file_provider import FileMarketData
from hyesg.market_data.snapshot import MarketDataSnapshot
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve

FIXTURES = Path(__file__).parent / "fixtures"


# ── Direct construction ──────────────────────────────────────


class TestSnapshotConstruction:
    """Tests for creating snapshots directly."""

    def test_empty_snapshot(self) -> None:
        snap = MarketDataSnapshot()
        assert snap.zero_curves == {}
        assert snap.inflation_curves == {}
        assert snap.credit_curves == {}
        assert snap.fx_spots == {}
        assert snap.equity_indices == {}

    def test_with_data(self) -> None:
        curve = ConstantCurve(0.04)
        snap = MarketDataSnapshot(
            zero_curves={"GBP": curve},
            fx_spots={"GBPUSD": 1.27},
        )
        assert "GBP" in snap.zero_curves
        assert snap.fx_spots["GBPUSD"] == 1.27

    def test_frozen(self) -> None:
        snap = MarketDataSnapshot()
        with pytest.raises(AttributeError):
            snap.fx_spots = {"GBPUSD": 1.30}  # type: ignore[misc]

    def test_currencies_property(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={
                "USD": ConstantCurve(0.04),
                "GBP": ConstantCurve(0.05),
            }
        )
        assert snap.currencies == ["GBP", "USD"]

    def test_ratings_property(self) -> None:
        snap = MarketDataSnapshot(
            credit_curves={
                "BBB": {"GBP": ConstantCurve(0.01)},
                "AAA": {"GBP": ConstantCurve(0.005)},
            }
        )
        assert snap.ratings == ["AAA", "BBB"]


# ── from_provider ────────────────────────────────────────────


class TestSnapshotFromProvider:
    """Tests for building snapshots from a MarketDataProvider."""

    def test_loads_zero_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, currencies=["GBP"]
        )
        assert "GBP" in snap.zero_curves
        assert isinstance(snap.zero_curves["GBP"], ParametricCurve)

    def test_loads_inflation_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, currencies=["GBP"]
        )
        assert "GBP" in snap.inflation_curves

    def test_loads_credit_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider,
            credit_ratings=["AAA"],
            credit_currencies=["GBP"],
        )
        assert "AAA" in snap.credit_curves
        assert "GBP" in snap.credit_curves["AAA"]

    def test_loads_fx_spot(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, fx_pairs=[("GBP", "USD")]
        )
        assert "GBPUSD" in snap.fx_spots
        assert abs(snap.fx_spots["GBPUSD"] - 1.2735) < 1e-10

    def test_loads_equity_index(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, equity_names=["FTSE100"]
        )
        assert "FTSE100" in snap.equity_indices
        assert abs(snap.equity_indices["FTSE100"] - 7733.24) < 1e-10

    def test_missing_currency_skipped(self) -> None:
        """Missing data is logged and skipped, not raised."""
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, currencies=["EUR"]
        )
        assert "EUR" not in snap.zero_curves

    def test_missing_fx_pair_skipped(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider, fx_pairs=[("GBP", "JPY")]
        )
        assert "GBPJPY" not in snap.fx_spots

    def test_empty_request(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(provider)
        assert snap.zero_curves == {}
        assert snap.fx_spots == {}

    def test_full_load(self) -> None:
        """Load everything available in fixtures."""
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider,
            currencies=["GBP"],
            credit_ratings=["AAA"],
            credit_currencies=["GBP"],
            fx_pairs=[("GBP", "USD")],
            equity_names=["FTSE100"],
        )
        assert len(snap.zero_curves) == 1
        assert len(snap.inflation_curves) == 1
        assert len(snap.credit_curves) == 1
        assert len(snap.fx_spots) == 1
        assert len(snap.equity_indices) == 1

    def test_date_parameter_passed_through(self) -> None:
        """Date parameter should be forwarded (no error)."""
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider,
            currencies=["GBP"],
            date="2024-01-01",
        )
        assert "GBP" in snap.zero_curves
