"""Tests for AllocationRebalancer."""

from __future__ import annotations

import pytest

from hyesg.models.portfolio.rebalancer import AllocationRebalancer


class TestAllocationRebalancer:
    """Tests for the portfolio rebalancer."""

    def setup_method(self) -> None:
        """Create a rebalancer for each test."""
        self.rebalancer = AllocationRebalancer()

    def test_already_at_target_zero_trades(self) -> None:
        """If current allocation matches target, all trades are zero."""
        current = {"equity": 60.0, "bonds": 40.0}
        target = {"equity": 0.6, "bonds": 0.4}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == pytest.approx(0.0, abs=1e-5)
        assert trades["bonds"] == pytest.approx(0.0, abs=1e-5)

    def test_simple_two_asset_rebalance(self) -> None:
        """Buy/sell to reach target from an imbalanced position."""
        current = {"equity": 70.0, "bonds": 30.0}
        target = {"equity": 0.5, "bonds": 0.5}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == pytest.approx(-20.0, abs=1e-5)
        assert trades["bonds"] == pytest.approx(20.0, abs=1e-5)

    def test_three_asset_rebalance(self) -> None:
        """Three-asset rebalance sums to zero net trade."""
        current = {"A": 50.0, "B": 30.0, "C": 20.0}
        target = {"A": 0.4, "B": 0.4, "C": 0.2}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        net_trade = sum(trades.values())
        assert net_trade == pytest.approx(0.0, abs=1e-5)

    def test_min_transaction_threshold(self) -> None:
        """Trades below min_transaction threshold are suppressed to zero."""
        current = {"equity": 50.0000001, "bonds": 49.9999999}
        target = {"equity": 0.5, "bonds": 0.5}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == 0.0
        assert trades["bonds"] == 0.0

    def test_short_detection_raises(self) -> None:
        """Short positions are rejected."""
        current = {"equity": -1.0, "bonds": 101.0}
        target = {"equity": 0.5, "bonds": 0.5}
        with pytest.raises(ValueError, match="Short position"):
            self.rebalancer.rebalance(current, target, total_value=100.0)

    def test_zero_total_value_raises(self) -> None:
        """Zero total value raises ValueError."""
        current = {"equity": 0.0}
        target = {"equity": 1.0}
        with pytest.raises(ValueError, match="below minimum"):
            self.rebalancer.rebalance(current, target, total_value=0.0)

    def test_new_asset_in_target(self) -> None:
        """Asset in target but not in current → buy."""
        current = {"equity": 100.0}
        target = {"equity": 0.5, "bonds": 0.5}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == pytest.approx(-50.0, abs=1e-5)
        assert trades["bonds"] == pytest.approx(50.0, abs=1e-5)

    def test_asset_removed_from_target(self) -> None:
        """Asset in current but not in target → sell all."""
        current = {"equity": 50.0, "bonds": 50.0}
        target = {"equity": 1.0}
        trades = self.rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == pytest.approx(50.0, abs=1e-5)
        assert trades["bonds"] == pytest.approx(-50.0, abs=1e-5)

    def test_custom_tolerance(self) -> None:
        """Rebalancer accepts custom tolerance parameter."""
        rebalancer = AllocationRebalancer(tol=1e-12, max_iter=200)
        current = {"equity": 60.0, "bonds": 40.0}
        target = {"equity": 0.6, "bonds": 0.4}
        trades = rebalancer.rebalance(current, target, total_value=100.0)
        assert trades["equity"] == pytest.approx(0.0, abs=1e-5)
