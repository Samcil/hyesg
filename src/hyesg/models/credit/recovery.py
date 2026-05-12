"""Recovery strategies for credit bond defaults.

Defines the ``RecoveryStrategy`` protocol and four concrete
implementations:

- ``FaceValueRecovery``: recovery = R × face value
- ``MarketValueRecovery``: recovery = R × market value at default
- ``TreasuryValueRecovery``: recovery = R × risk-free equivalent value
- ``NoRecovery``: zero recovery (null object pattern)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol for bond recovery on default."""

    def recovery_value(
        self,
        face_value: Array,
        market_value: Array,
        risk_free_value: Array,
    ) -> Array:
        """Calculate recovery amount on default.

        Args:
            face_value: Face (par) value of the bond.
            market_value: Market value at default.
            risk_free_value: Risk-free equivalent (treasury) value.

        Returns:
            Recovery amount.
        """
        ...


class FaceValueRecovery:
    """Recovery = R × FaceValue.

    The most common assumption in reduced-form credit models.

    Args:
        recovery_rate: Fraction of face value recovered on default.
    """

    def __init__(self, recovery_rate: float) -> None:
        self.recovery_rate = recovery_rate

    def recovery_value(
        self,
        face_value: Array,
        market_value: Array,
        risk_free_value: Array,
    ) -> Array:
        """Return recovery_rate × face_value.

        Args:
            face_value: Face (par) value of the bond.
            market_value: Market value at default (unused).
            risk_free_value: Risk-free equivalent value (unused).

        Returns:
            Recovery amount.
        """
        return self.recovery_rate * face_value


class MarketValueRecovery:
    """Recovery = R × MarketValue at default.

    Args:
        recovery_rate: Fraction of market value recovered on default.
    """

    def __init__(self, recovery_rate: float) -> None:
        self.recovery_rate = recovery_rate

    def recovery_value(
        self,
        face_value: Array,
        market_value: Array,
        risk_free_value: Array,
    ) -> Array:
        """Return recovery_rate × market_value.

        Args:
            face_value: Face (par) value (unused).
            market_value: Market value at default.
            risk_free_value: Risk-free equivalent value (unused).

        Returns:
            Recovery amount.
        """
        return self.recovery_rate * market_value


class TreasuryValueRecovery:
    """Recovery = R × risk-free equivalent value.

    Args:
        recovery_rate: Fraction of treasury value recovered on default.
    """

    def __init__(self, recovery_rate: float) -> None:
        self.recovery_rate = recovery_rate

    def recovery_value(
        self,
        face_value: Array,
        market_value: Array,
        risk_free_value: Array,
    ) -> Array:
        """Return recovery_rate × risk_free_value.

        Args:
            face_value: Face (par) value (unused).
            market_value: Market value at default (unused).
            risk_free_value: Risk-free equivalent (treasury) value.

        Returns:
            Recovery amount.
        """
        return self.recovery_rate * risk_free_value


class NoRecovery:
    """Zero recovery on default (null object pattern)."""

    def recovery_value(
        self,
        face_value: Array,
        market_value: Array,
        risk_free_value: Array,
    ) -> Array:
        """Return zero recovery.

        Args:
            face_value: Face (par) value (unused).
            market_value: Market value at default (unused).
            risk_free_value: Risk-free equivalent value (unused).

        Returns:
            Array of zeros matching face_value shape.
        """
        return jnp.zeros_like(face_value)
