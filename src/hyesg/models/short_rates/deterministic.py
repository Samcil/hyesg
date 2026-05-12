"""Deterministic short rate model (null-object pattern).

The short rate path is entirely determined by the initial yield curve:
    r(t) = f(0, t)  (instantaneous forward rate from market curve)

This model:
- Has ZERO volatility (no stochastic component)
- Requires ZERO Brownian shocks
- Produces EXACT discount factors from the market curve
- Acts as a null-object: safe to wire into dependency graphs
  where a ShortRateModel is required but stochastic behavior is not

Used when:
- Testing pricing formulas against exact market values
- Providing a baseline/benchmark for stochastic models
- Foreign economies where only deterministic rates are needed
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import OUState, ShockConfig
from hyesg.math.curves.protocol import ParametricCurve


@register_model("deterministic")
class Deterministic:
    """Deterministic short rate model with zero volatility.

    Reads forward rates from a pre-supplied market curve, producing
    exact (non-stochastic) short rate paths.  Implements the full
    ``ShortRateModel`` protocol so it can substitute for any
    stochastic short rate model.

    Args:
        forward_curve: Market forward rate curve f(0, t).
        name: Unique model name.
    """

    def __init__(
        self,
        forward_curve: ParametricCurve,
        name: str = "deterministic",
    ) -> None:
        self._forward_curve = forward_curve
        self._name = name

    # ─── Model metadata ───

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def n_shocks(self) -> int:
        """Number of independent Brownian increments (zero)."""
        return 0

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata — no shocks required."""
        return ShockConfig(
            n_shocks=0,
            distribution="normal",
            correlate=False,
            names=(),
        )

    # ─── State management ───

    def init_state(self, params: Any = None, market: Any = None) -> OUState:
        """Create initial state from the forward curve at t=0.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial OUState with x=0 and short_rate=f(0).
        """
        r0 = self._forward_curve.evaluate(0.0)
        return OUState(
            x=jnp.array(0.0, dtype=jnp.float64),
            short_rate=jnp.array(r0, dtype=jnp.float64),
        )

    def step(
        self,
        state: OUState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[OUState, dict[str, Any]]:
        """Advance by reading the next forward rate from the curve.

        Ignores shocks entirely.  The forward curve is a pure-Python
        object evaluated at trace time — the resulting constant is
        embedded in the XLA computation, which is safe because the
        time grid is fixed before JIT compilation.

        Args:
            state: Current OUState (unused beyond carry semantics).
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Ignored (model requires zero shocks).
            deps: Dependency outputs (unused).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        next_rate = self._forward_curve.evaluate(float(t + dt))
        short_rate = jnp.array(next_rate, dtype=jnp.float64)
        new_state = OUState(
            x=jnp.array(0.0, dtype=jnp.float64),
            short_rate=short_rate,
        )
        return new_state, {"short_rate": short_rate}

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: OUState) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current OUState.

        Returns:
            Short rate value.
        """
        return state.short_rate

    def zcb_price(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T) from the market curve.

        Uses the exact relationship:
            P(t, T) = P(0, T) / P(0, t)

        where P(0, s) = exp(-∫₀ˢ f(0, u) du).

        Args:
            state: Current OUState (unused — pricing is deterministic).
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        tau = maturity - t
        if abs(tau) < 1e-12:
            return jnp.array(1.0, dtype=jnp.float64)
        integral_0_t = self._forward_curve.integral(0.0, float(t))
        integral_0_T = self._forward_curve.integral(0.0, float(maturity))
        log_price = -(integral_0_T - integral_0_t)
        return jnp.array(math.exp(log_price), dtype=jnp.float64)

    def spot_rate(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Continuously compounded spot rate R(t, T).

        R(t, T) = -ln P(t, T) / (T - t)

        Args:
            state: Current OUState.
            t: Current time.
            maturity: Maturity time.

        Returns:
            Spot rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, maturity)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)

    def forward_rate(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Instantaneous forward rate f(t, T).

        For the deterministic model this equals the market forward rate
        at maturity T, since f(t, T) = f(0, T) when rates are
        non-stochastic.

        Args:
            state: Current OUState (unused).
            t: Current time (unused for deterministic curve).
            maturity: Forward time.

        Returns:
            Instantaneous forward rate.
        """
        return jnp.array(
            self._forward_curve.evaluate(float(maturity)),
            dtype=jnp.float64,
        )

    def swap_rate(
        self,
        state: OUState,
        t: float,
        tenor: float,
        freq: float,
    ) -> jnp.ndarray:
        """Par swap rate S(t; tenor, freq).

        S = (1 - P(t, t+tenor)) / Σ_{i=1}^{n} freq · P(t, t + i·freq)

        Args:
            state: Current OUState.
            t: Current time.
            tenor: Swap tenor in years.
            freq: Payment frequency in years (e.g. 0.5 for semi-annual).

        Returns:
            Par swap rate.
        """
        n_payments = int(round(tenor / freq))
        annuity = jnp.array(0.0, dtype=jnp.float64)
        for i in range(1, n_payments + 1):
            annuity = annuity + freq * self.zcb_price(state, t, t + i * freq)
        numerator = 1.0 - self.zcb_price(state, t, t + tenor)
        return numerator / jnp.maximum(annuity, 1e-12)
