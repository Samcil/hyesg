"""Credit Default Swap (CDS) pricing.

A CDS provides credit protection on a reference entity.
The protection buyer pays a periodic spread; the protection
seller pays the loss given default if a credit event occurs.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class CDS:
    """Credit Default Swap.

    The CDS value is computed as the difference between the
    protection leg (expected loss) and the premium leg
    (discounted spread payments weighted by survival probability).

    Attributes:
        reference_label: Identifier for the reference entity.
        spread: Annual CDS spread (premium rate).
        notional: Notional principal.
        maturity: Maturity in years.
    """

    def __init__(
        self,
        reference_label: str,
        spread: float,
        notional: float,
        maturity: float,
    ) -> None:
        """Initialise CDS.

        Args:
            reference_label: Name/label of the reference entity.
            spread: Annual CDS spread.
            notional: Notional principal.
            maturity: Maturity in years.
        """
        self.reference_label = reference_label
        self.spread = spread
        self.notional = notional
        self.maturity = maturity

    def value(
        self,
        survival_prob: Array,
        discount_factors: Array,
        t: float,
        recovery_rate: float = 0.4,
    ) -> Array:
        """Compute CDS mark-to-market value (from protection buyer perspective).

        Premium leg: sum of spread * dt * survival_prob * df
        Protection leg: sum of (1-R) * default_prob * df

        MTM = protection_leg - premium_leg

        At inception (fair spread), MTM = 0.

        Args:
            survival_prob: Survival probabilities at each period, shape (n_periods,).
            discount_factors: Discount factors at each period, shape (n_periods,).
            t: Current valuation time.
            recovery_rate: Recovery rate on default (default 0.4 = 40%).

        Returns:
            Mark-to-market value of the CDS.
        """
        n_periods = survival_prob.shape[0]
        dt = (self.maturity - t) / n_periods

        # Premium leg: spread * sum(survival * df * dt)
        premium_leg = self.spread * jnp.sum(survival_prob * discount_factors * dt)

        # Default probabilities: marginal default in each period
        surv_shifted = jnp.concatenate([jnp.array([1.0]), survival_prob[:-1]])
        default_prob = surv_shifted - survival_prob

        # Protection leg: (1-R) * sum(default_prob * df)
        protection_leg = (1.0 - recovery_rate) * jnp.sum(
            default_prob * discount_factors
        )

        return self.notional * (protection_leg - premium_leg)
