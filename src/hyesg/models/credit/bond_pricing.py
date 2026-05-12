"""Coupon bond and index-linked bond pricing.

Survival-weighted discounted cashflow pricing for credit-risky bonds.
Supports semi-annual coupon bonds and index-linked bonds combining
credit and inflation adjustments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


@runtime_checkable
class BondPricer(Protocol):
    """Protocol for bond pricing implementations."""

    def price(
        self,
        coupon_rate: float,
        maturity: float,
        survival_fn: Callable[[float], Array],
        zcb_fn: Callable[[float], Array],
        recovery_rate: float,
        t: float,
        inflation_index_fn: Callable[[float], Array] | None = None,
    ) -> Array:
        """Price a credit-risky bond.

        Args:
            coupon_rate: Annual coupon rate.
            maturity: Bond maturity in years from t=0.
            survival_fn: S(T) -> survival probability to time T.
            zcb_fn: P(T) -> risk-free ZCB price to time T.
            recovery_rate: Recovery fraction on default.
            t: Current valuation time.
            inflation_index_fn: Optional I(T) -> inflation index level.
                Required for index-linked bonds, ignored for nominal.

        Returns:
            Bond price.
        """
        ...


class CouponBondPricer:
    """Semi-annual coupon bond pricing using survival-weighted cashflows.

    The price is computed as:
        V = Σ_i c/2 * P(t_i) * S(t_i) + P(T) * S(T) + R * Σ_i P(t_i) * dPD(t_i)

    where:
    - c is the annual coupon rate
    - P(t_i) is the risk-free ZCB price
    - S(t_i) is the survival probability
    - R is the recovery rate
    - dPD(t_i) is the incremental default probability

    Args:
        frequency: Coupon payment frequency per year (default 2 = semi-annual).
    """

    def __init__(self, frequency: int = 2) -> None:
        self._frequency = frequency

    def price(
        self,
        coupon_rate: float,
        maturity: float,
        survival_fn: Callable[[float], Array],
        zcb_fn: Callable[[float], Array],
        recovery_rate: float,
        t: float,
    ) -> Array:
        """Price a semi-annual coupon bond with credit risk.

        Args:
            coupon_rate: Annual coupon rate (e.g. 0.05 for 5%).
            maturity: Bond maturity in years from t=0.
            survival_fn: S(T) -> survival probability from t to T.
            zcb_fn: P(T) -> risk-free ZCB price from t to T.
            recovery_rate: Recovery fraction on default.
            t: Current valuation time.

        Returns:
            Bond dirty price.
        """
        remaining = maturity - t
        if remaining <= 0:
            return jnp.array(1.0, dtype=jnp.float64)

        dt_coupon = 1.0 / self._frequency
        coupon_per_period = coupon_rate / self._frequency

        # Build coupon payment times
        n_periods = max(1, int(remaining * self._frequency))
        coupon_times = [t + (i + 1) * dt_coupon for i in range(n_periods)]
        # Ensure last coupon at maturity
        if coupon_times and coupon_times[-1] > maturity:
            coupon_times[-1] = maturity

        # Coupon leg: Σ c/freq * P(t_i) * S(t_i)
        coupon_pv = jnp.array(0.0, dtype=jnp.float64)
        for t_i in coupon_times:
            coupon_pv = coupon_pv + coupon_per_period * zcb_fn(t_i) * survival_fn(t_i)

        # Principal leg: P(T) * S(T)
        principal_pv = zcb_fn(maturity) * survival_fn(maturity)

        # Recovery leg: R * Σ P(t_i) * [S(t_{i-1}) - S(t_i)]
        recovery_pv = jnp.array(0.0, dtype=jnp.float64)
        prev_surv = survival_fn(t)
        for t_i in coupon_times:
            curr_surv = survival_fn(t_i)
            dpd = prev_surv - curr_surv  # incremental default probability
            recovery_pv = recovery_pv + recovery_rate * zcb_fn(t_i) * dpd
            prev_surv = curr_surv

        return coupon_pv + principal_pv + recovery_pv


class IndexLinkedBondPricer:
    """Index-linked credit bond pricing combining credit and inflation.

    Similar to ``CouponBondPricer`` but cashflows are adjusted by an
    inflation index ratio, producing real-return credit bonds.

    Args:
        frequency: Coupon payment frequency per year (default 2).
    """

    def __init__(self, frequency: int = 2) -> None:
        self._frequency = frequency

    def price(
        self,
        coupon_rate: float,
        maturity: float,
        survival_fn: Callable[[float], Array],
        zcb_fn: Callable[[float], Array],
        recovery_rate: float,
        t: float,
        inflation_index_fn: Callable[[float], Array] | None = None,
    ) -> Array:
        """Price an index-linked credit bond.

        Cashflows are inflation-adjusted:
            V = Σ c/freq * I(t_i)/I(t) * P(t_i) * S(t_i)
                + I(T)/I(t) * P(T) * S(T)
                + R * Σ I(t_i)/I(t) * P(t_i) * dPD(t_i)

        Args:
            coupon_rate: Annual real coupon rate.
            maturity: Bond maturity in years from t=0.
            survival_fn: S(T) -> survival probability.
            zcb_fn: P(T) -> risk-free ZCB price.
            recovery_rate: Recovery fraction on default.
            t: Current valuation time.
            inflation_index_fn: I(T) -> inflation index level at T.
                Required for index-linked bonds.

        Returns:
            Index-linked bond price.

        Raises:
            ValueError: If inflation_index_fn is None.
        """
        if inflation_index_fn is None:
            raise ValueError(
                "inflation_index_fn is required for IndexLinkedBondPricer"
            )
        remaining = maturity - t
        if remaining <= 0:
            return jnp.array(1.0, dtype=jnp.float64)

        dt_coupon = 1.0 / self._frequency
        coupon_per_period = coupon_rate / self._frequency

        n_periods = max(1, int(remaining * self._frequency))
        coupon_times = [t + (i + 1) * dt_coupon for i in range(n_periods)]
        if coupon_times and coupon_times[-1] > maturity:
            coupon_times[-1] = maturity

        base_index = inflation_index_fn(t)

        # Coupon leg with inflation adjustment
        coupon_pv = jnp.array(0.0, dtype=jnp.float64)
        for t_i in coupon_times:
            index_ratio = inflation_index_fn(t_i) / jnp.maximum(base_index, 1e-12)
            coupon_pv = (
                coupon_pv
                + coupon_per_period * index_ratio * zcb_fn(t_i) * survival_fn(t_i)
            )

        # Principal leg with inflation
        index_ratio_T = inflation_index_fn(maturity) / jnp.maximum(base_index, 1e-12)
        principal_pv = index_ratio_T * zcb_fn(maturity) * survival_fn(maturity)

        # Recovery leg with inflation
        recovery_pv = jnp.array(0.0, dtype=jnp.float64)
        prev_surv = survival_fn(t)
        for t_i in coupon_times:
            curr_surv = survival_fn(t_i)
            dpd = prev_surv - curr_surv
            index_ratio_i = inflation_index_fn(t_i) / jnp.maximum(base_index, 1e-12)
            recovery_pv = (
                recovery_pv + recovery_rate * index_ratio_i * zcb_fn(t_i) * dpd
            )
            prev_surv = curr_surv

        return coupon_pv + principal_pv + recovery_pv
