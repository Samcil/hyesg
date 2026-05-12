"""Tests for workload distribution across devices."""

from __future__ import annotations

import pytest

from hyesg.engine.workload import distribute_trials


class TestDistributeTrials:
    """Tests for distribute_trials."""

    def test_even_distribution(self) -> None:
        """Trials divide evenly across devices."""
        result = distribute_trials(100, 4, require_even=False)
        assert result == (25, 25, 25, 25)

    def test_uneven_with_remainder(self) -> None:
        """Remainder goes to the last device."""
        result = distribute_trials(103, 4, require_even=False)
        # base = 25, remainder = 103 - 25*3 = 28
        assert result == (25, 25, 25, 28)

    def test_require_even_rounds_up(self) -> None:
        """Odd base is rounded up to even for antithetic pairing."""
        # 103 / 4 = 25 (odd) → rounds up to 26
        result = distribute_trials(103, 4, require_even=True)
        assert all(r % 2 == 0 for r in result)

    def test_single_device_gets_all(self) -> None:
        """Single device gets all trials."""
        result = distribute_trials(100, 1, require_even=False)
        assert result == (100,)

    def test_single_device_even(self) -> None:
        """Single device with odd trials gets rounded up."""
        result = distribute_trials(99, 1, require_even=True)
        assert result[0] % 2 == 0
        assert result[0] >= 99

    def test_total_ge_requested(self) -> None:
        """Sum of distributed trials ≥ total (may round up)."""
        result = distribute_trials(101, 3, require_even=True)
        assert sum(result) >= 101

    def test_total_preserved_no_rounding(self) -> None:
        """Without rounding, sum exactly equals total."""
        result = distribute_trials(100, 4, require_even=False)
        assert sum(result) == 100

    def test_two_devices(self) -> None:
        """Two devices split evenly."""
        result = distribute_trials(200, 2, require_even=True)
        assert len(result) == 2
        assert all(r % 2 == 0 for r in result)
        assert sum(result) >= 200

    def test_many_devices(self) -> None:
        """Many devices with few trials each."""
        result = distribute_trials(20, 10, require_even=False)
        assert len(result) == 10
        assert sum(result) == 20

    def test_invalid_total_raises(self) -> None:
        """Zero total_trials raises ValueError."""
        with pytest.raises(ValueError, match="total_trials"):
            distribute_trials(0, 4)

    def test_invalid_devices_raises(self) -> None:
        """Zero devices raises ValueError."""
        with pytest.raises(ValueError, match="n_devices"):
            distribute_trials(100, 0)
