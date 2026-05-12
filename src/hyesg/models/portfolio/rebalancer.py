"""Portfolio rebalancer — compute trades to achieve target allocation.

Uses a simple proportional rebalancing approach with thresholds matching
the C# ESG engine conventions.
"""

from __future__ import annotations

# C# threshold constants
_MIN_TRANSACTION = 1e-6
_MIN_VALUE = 1e-8
_SHORT_DETECTION = 1e-12


class AllocationRebalancer:
    """Rebalance a portfolio to target weights.

    Computes the trades required to move from current holdings to a
    target allocation, respecting minimum transaction and value
    thresholds.

    Thresholds from C#:
        - min_transaction: 1e-6
        - min_value: 1e-8
        - short_detection: 1e-12

    Args:
        tol: Convergence tolerance for weight matching.
        max_iter: Maximum iterations (unused in proportional mode;
            reserved for Brent root-finding extension).
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def rebalance(
        self,
        current_values: dict[str, float],
        target_weights: dict[str, float],
        total_value: float,
    ) -> dict[str, float]:
        """Compute trades to achieve the target allocation.

        Args:
            current_values: Current market value per asset.
            target_weights: Target weight per asset (should sum to ~1).
            total_value: Total portfolio value to allocate across assets.

        Returns:
            Dict of ``asset -> trade_amount``.  Positive means buy,
            negative means sell.

        Raises:
            ValueError: If *total_value* is non-positive or a short
                position is detected.
        """
        if total_value < _MIN_VALUE:
            raise ValueError(
                f"Total portfolio value {total_value} is below minimum "
                f"threshold {_MIN_VALUE}"
            )

        # Check for short positions
        for asset, val in current_values.items():
            if val < -_SHORT_DETECTION:
                raise ValueError(
                    f"Short position detected in '{asset}': {val}"
                )

        trades: dict[str, float] = {}
        for asset, target_w in target_weights.items():
            target_val = target_w * total_value
            current_val = current_values.get(asset, 0.0)
            trade = target_val - current_val

            # Suppress tiny trades below minimum transaction threshold
            if abs(trade) < _MIN_TRANSACTION:
                trade = 0.0

            trades[asset] = trade

        # Handle assets in current_values but not in target_weights (sell all)
        for asset in current_values:
            if asset not in target_weights:
                sell_amount = -current_values[asset]
                if abs(sell_amount) >= _MIN_TRANSACTION:
                    trades[asset] = sell_amount

        return trades
