"""Expected loss (P1) calculation for regulatory capital.

Implements the standard formula:
    P1 = EAD × (1 - R) × PD

where EAD is exposure at default, R is recovery rate, and PD is
the probability of default.
"""

from __future__ import annotations

from jax import Array

from hyesg.models.credit.recovery import RecoveryStrategy


class P1Calculator:
    """Expected loss calculation for regulatory capital.

    P1 = EAD × (1 - R) × PD

    where:
    - EAD = exposure at default
    - R = recovery rate (derived from the recovery strategy)
    - PD = probability of default = 1 - survival_prob

    Args:
        recovery_strategy: Strategy for computing recovery on default.
    """

    def __init__(self, recovery_strategy: RecoveryStrategy) -> None:
        self.recovery_strategy = recovery_strategy

    def expected_loss(
        self,
        exposure: Array,
        survival_prob: Array,
    ) -> Array:
        """Calculate expected loss from exposure and survival probability.

        Args:
            exposure: Exposure at default (EAD).
            survival_prob: Survival probability S(t).

        Returns:
            Expected loss = (exposure - recovery) × PD.
        """
        pd = 1.0 - survival_prob
        recovery = self.recovery_strategy.recovery_value(
            exposure, exposure, exposure
        )
        return (exposure - recovery) * pd
