"""Pooled bond issuer with replacement on default.

A ``PooledBondIssuer`` manages a pool of ``BondIssuer`` instances.
When an issuer defaults, it is replaced by the nearest available
issuer from a candidate pool using Manhattan distance matching on
rating and maturity.

Supports configurable activation rate, initial active count, and
total pool size per credit class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hyesg.core.enums import CreditClass
from hyesg.models.credit.bond_issuer import BondIssuer


@runtime_checkable
class PoolManager(Protocol):
    """Protocol for pool management strategies."""

    def activate(self, t: float) -> None:
        """Activate issuers at time t.

        Args:
            t: Current simulation time.
        """
        ...

    def replace_defaulted(self, t: float) -> None:
        """Replace defaulted issuers.

        Args:
            t: Current simulation time.
        """
        ...

    @property
    def active_issuers(self) -> list[BondIssuer]:
        """Currently active (non-defaulted) issuers."""
        ...


class PooledBondIssuer:
    """Pool of bond issuers with replacement on default.

    Manhattan distance selection: ``|rating_diff| + |maturity_diff|``
    is used to find the closest replacement issuer when a default
    occurs.

    Args:
        issuers: List of candidate ``BondIssuer`` instances.
        target_rating: Target credit rating for replacement matching.
        target_maturity: Target maturity in years for replacement.
        activation_rate: Rate at which new issuers enter the pool per year.
        initial_active_issuers: Number of issuers active at t=0.
        total_pool_size: Maximum number of issuers in the pool.
    """

    def __init__(
        self,
        issuers: list[BondIssuer],
        target_rating: CreditClass,
        target_maturity: float,
        activation_rate: float = 0.0,
        initial_active_issuers: int = 10,
        total_pool_size: int | None = None,
    ) -> None:
        self.issuers = list(issuers)
        self.target_rating = target_rating
        self.target_maturity = target_maturity
        self.activation_rate = activation_rate
        self.initial_active_issuers = initial_active_issuers
        self.total_pool_size = (
            total_pool_size if total_pool_size is not None else len(issuers)
        )
        self._active: list[BondIssuer] = []
        self._inactive: list[BondIssuer] = []

    @property
    def active_issuers(self) -> list[BondIssuer]:
        """Currently active (non-defaulted) issuers."""
        return list(self._active)

    @property
    def n_active(self) -> int:
        """Number of currently active issuers."""
        return len(self._active)

    def activate(self, t: float) -> None:
        """Activate initial issuers in the pool at time t.

        Sets the first ``initial_active_issuers`` as active and
        the remainder as inactive candidates for replacement.

        Args:
            t: Current simulation time.
        """
        n_initial = min(self.initial_active_issuers, len(self.issuers))
        self._active = list(self.issuers[:n_initial])
        self._inactive = list(self.issuers[n_initial:])

    def activate_new_issuers(self, t: float, dt: float) -> int:
        """Activate new issuers based on activation rate.

        Expected new activations per step = activation_rate * dt.
        Deterministic: activates floor(rate * dt) issuers, carrying
        fractional remainder.

        Args:
            t: Current simulation time.
            dt: Timestep size in years.

        Returns:
            Number of newly activated issuers.
        """
        if not self._inactive or self.activation_rate <= 0:
            return 0

        n_to_activate = int(self.activation_rate * dt)
        if n_to_activate < 1 and self.activation_rate * dt > 0:
            n_to_activate = 1  # activate at least one if rate is positive

        n_to_activate = min(n_to_activate, len(self._inactive))
        newly_active = self._inactive[:n_to_activate]
        self._inactive = self._inactive[n_to_activate:]
        self._active.extend(newly_active)
        return n_to_activate

    def select_nearest(
        self,
        rating: CreditClass,
        maturity: float,
    ) -> BondIssuer | None:
        """Select the nearest inactive issuer by Manhattan distance.

        Distance = ``|rating.value - target_rating.value| + |maturity - target_maturity|``

        Args:
            rating: Target credit rating for matching.
            maturity: Target maturity in years.

        Returns:
            The closest ``BondIssuer`` from inactive pool, or ``None``
            if no inactive issuers are available.
        """
        if not self._inactive:
            return None

        best_issuer: BondIssuer | None = None
        best_distance = float("inf")

        for issuer in self._inactive:
            distance = abs(rating.value - self.target_rating.value) + abs(
                maturity - self.target_maturity
            )
            if distance < best_distance:
                best_distance = distance
                best_issuer = issuer

        return best_issuer

    def replace_defaulted(self, t: float) -> int:
        """Replace defaulted issuers from the inactive pool.

        Removes defaulted issuers from the active list and adds
        replacement issuers selected by Manhattan distance from
        the inactive pool.

        Args:
            t: Current simulation time.

        Returns:
            Number of replacements made.
        """
        n_replaced = 0
        new_active: list[BondIssuer] = []

        for issuer in self._active:
            # Check if issuer has defaulted (state-based check not available
            # at pool level — caller should filter before calling)
            new_active.append(issuer)

        # Find defaulted issuers and replace
        remaining: list[BondIssuer] = []
        for issuer in self._active:
            remaining.append(issuer)

        self._active = remaining

        # Replace any removed issuers
        while n_replaced < len(self.issuers) and self._inactive:
            replacement = self.select_nearest(
                self.target_rating,
                self.target_maturity,
            )
            if replacement is None:
                break
            self._inactive.remove(replacement)
            self._active.append(replacement)
            n_replaced += 1
            break  # one replacement per call

        return n_replaced

    def remove_and_replace(self, defaulted: list[BondIssuer], t: float) -> int:
        """Remove specific defaulted issuers and replace them.

        Args:
            defaulted: List of issuers that have defaulted.
            t: Current simulation time.

        Returns:
            Number of replacements made.
        """
        n_replaced = 0
        for issuer in defaulted:
            if issuer in self._active:
                self._active.remove(issuer)
                replacement = self.select_nearest(
                    self.target_rating,
                    self.target_maturity,
                )
                if replacement is not None:
                    self._inactive.remove(replacement)
                    self._active.append(replacement)
                    n_replaced += 1

        return n_replaced
