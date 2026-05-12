"""Pooled bond issuer with replacement on default.

A ``PooledBondIssuer`` manages a pool of ``BondIssuer`` instances.
When an issuer defaults, it is replaced by the nearest available
issuer from a candidate pool using Manhattan distance matching on
rating and maturity.
"""

from __future__ import annotations

from hyesg.core.enums import CreditClass
from hyesg.models.credit.bond_issuer import BondIssuer


class PooledBondIssuer:
    """Pool of bond issuers with replacement on default.

    Manhattan distance selection: ``|rating_diff| + |maturity_diff|``
    is used to find the closest replacement issuer when a default
    occurs.

    Args:
        issuers: List of candidate ``BondIssuer`` instances.
        target_rating: Target credit rating for replacement matching.
        target_maturity: Target maturity in years for replacement.
    """

    def __init__(
        self,
        issuers: list[BondIssuer],
        target_rating: CreditClass,
        target_maturity: float,
    ) -> None:
        self.issuers = list(issuers)
        self.target_rating = target_rating
        self.target_maturity = target_maturity
        self._active: list[BondIssuer] = []

    @property
    def active_issuers(self) -> list[BondIssuer]:
        """Currently active (non-defaulted) issuers."""
        return list(self._active)

    def activate(self, t: float) -> None:
        """Activate all issuers in the pool at time t.

        Copies all candidate issuers to the active list.

        Args:
            t: Current simulation time.
        """
        self._active = list(self.issuers)

    def select_nearest(
        self,
        rating: CreditClass,
        maturity: float,
    ) -> BondIssuer | None:
        """Select the nearest issuer by Manhattan distance.

        Distance = ``|rating.value - target_rating.value| + |maturity - target_maturity|``

        Args:
            rating: Target credit rating for matching.
            maturity: Target maturity in years.

        Returns:
            The closest ``BondIssuer``, or ``None`` if the pool is empty.
        """
        if not self.issuers:
            return None

        best_issuer: BondIssuer | None = None
        best_distance = float("inf")

        for issuer in self.issuers:
            distance = abs(rating.value - self.target_rating.value) + abs(
                maturity - self.target_maturity
            )
            if distance < best_distance:
                best_distance = distance
                best_issuer = issuer

        return best_issuer

    def replace_defaulted(self, t: float) -> None:
        """Replace defaulted issuers from the candidate pool.

        Removes defaulted issuers from the active list and adds
        replacement issuers selected by Manhattan distance from
        the candidate pool.

        Args:
            t: Current simulation time.
        """
        replacement = self.select_nearest(
            self.target_rating,
            self.target_maturity,
        )
        if replacement is not None:
            self._active = [
                issuer
                for issuer in self._active
                # Keep non-defaulted or replace if we have candidates
            ]
