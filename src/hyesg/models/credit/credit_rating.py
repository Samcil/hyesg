"""Credit rating with dynamic re-rating capability.

A ``CreditRating`` wraps a ``CreditClass`` enum value and supports
dynamic updates based on intensity levels versus rating thresholds.
"""

from __future__ import annotations

from hyesg.core.enums import CreditClass


class CreditRating:
    """Credit rating with dynamic re-rating capability.

    Uses ``CreditClass`` enum (AAA=7 through Default=0).  The rating
    can be updated based on the current default intensity relative
    to a set of thresholds.

    Args:
        initial_rating: Starting credit rating.
    """

    def __init__(self, initial_rating: CreditClass) -> None:
        self._rating = initial_rating

    @property
    def rating(self) -> CreditClass:
        """Current credit rating."""
        return self._rating

    def update_rating(
        self,
        intensity: float,
        thresholds: dict[CreditClass, float],
    ) -> CreditClass:
        """Re-rate based on intensity level vs thresholds.

        Iterates from the lowest rating upward; the rating is set to
        the lowest class whose threshold the intensity exceeds.  If
        intensity is below all thresholds, the highest rating in the
        threshold dict (or the current rating) is assigned.

        Args:
            intensity: Current default intensity.
            thresholds: Mapping from ``CreditClass`` to intensity
                threshold.  An issuer is rated at the *lowest* class
                whose threshold is exceeded.

        Returns:
            Updated credit rating.
        """
        # Sort thresholds by rating value ascending (worst → best)
        sorted_ratings = sorted(thresholds.items(), key=lambda x: x[0].value)

        new_rating = self._rating
        for credit_class, threshold in sorted_ratings:
            if intensity >= threshold:
                new_rating = credit_class
                break

        # If intensity is below all thresholds, assign the best rating
        if intensity < min(thresholds.values()):
            best_rating = max(thresholds.keys(), key=lambda c: c.value)
            new_rating = best_rating

        self._rating = new_rating
        return self._rating
