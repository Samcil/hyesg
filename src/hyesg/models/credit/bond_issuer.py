"""Bond issuer with CIR++ intensity and Cox survival default monitoring.

A ``BondIssuer`` tracks a single issuer's default intensity, cumulative
intensity integral, survival probability, and default status.  Default
occurs at the first time the survival probability drops below a
pre-drawn uniform threshold (Cox process construction).

The master/slave pattern allows correlated defaults: the *master*
generates the uniform threshold from a PRNG key, while *slaves* reuse
a threshold supplied externally.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hyesg.models.credit.recovery import RecoveryStrategy


class IssuerState(NamedTuple):
    """State for a single bond issuer.

    Attributes:
        cum_intensity: Cumulative intensity integral ∫₀ᵗ λ(s) ds.
        survival_prob: Survival probability exp(-cum_intensity).
        uniform_threshold: Fixed uniform draw for default determination.
        has_defaulted: Boolean flag (1.0 = defaulted, 0.0 = alive).
        default_time: Time of default (inf if no default yet).
    """

    cum_intensity: Array
    survival_prob: Array
    uniform_threshold: Array
    has_defaulted: Array
    default_time: Array


class BondIssuer:
    """Single bond issuer with CIR++ intensity and default monitoring.

    Uses the master/slave pattern: a *master* issuer generates its own
    uniform threshold from a PRNG key; a *slave* reuses an externally
    supplied threshold for correlated defaults.

    Args:
        alpha: Mean-reversion speed for CIR intensity.
        sigma: Volatility of the CIR intensity process.
        initial_intensity: Starting default intensity λ(0).
        recovery_strategy: Strategy for computing recovery on default.
        is_master: If True, generate own uniform threshold; otherwise
            the threshold must be supplied via ``init_state_with_threshold``.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        initial_intensity: float,
        recovery_strategy: RecoveryStrategy,
        is_master: bool = True,
    ) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.initial_intensity = initial_intensity
        self.recovery_strategy = recovery_strategy
        self.is_master = is_master

    def init_state(self, key: Array | None = None) -> IssuerState:
        """Initialise issuer state with a random uniform threshold.

        Args:
            key: JAX PRNG key for drawing the uniform threshold.
                If None, a default key is created internally.

        Returns:
            Initial ``IssuerState`` with zero cumulative intensity,
            survival = 1, and no default.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        threshold = jax.random.uniform(key, dtype=jnp.float64)
        return IssuerState(
            cum_intensity=jnp.array(0.0, dtype=jnp.float64),
            survival_prob=jnp.array(1.0, dtype=jnp.float64),
            uniform_threshold=threshold,
            has_defaulted=jnp.array(0.0, dtype=jnp.float64),
            default_time=jnp.array(jnp.inf, dtype=jnp.float64),
        )

    def init_state_with_threshold(self, threshold: Array) -> IssuerState:
        """Initialise issuer state with an externally supplied threshold.

        Used by *slave* issuers to share a master's uniform draw for
        correlated defaults.

        Args:
            threshold: Pre-drawn uniform threshold in [0, 1].

        Returns:
            Initial ``IssuerState`` using the given threshold.
        """
        return IssuerState(
            cum_intensity=jnp.array(0.0, dtype=jnp.float64),
            survival_prob=jnp.array(1.0, dtype=jnp.float64),
            uniform_threshold=jnp.asarray(threshold, dtype=jnp.float64),
            has_defaulted=jnp.array(0.0, dtype=jnp.float64),
            default_time=jnp.array(jnp.inf, dtype=jnp.float64),
        )

    def update(
        self,
        state: IssuerState,
        intensity_increment: Array,
        dt: float,
        t: float,
    ) -> IssuerState:
        """Update survival probability and check for default.

        Default occurs when survival_prob drops below uniform_threshold.
        Once defaulted, the flag stays set and default_time is frozen.

        Args:
            state: Current issuer state.
            intensity_increment: Current intensity λ(t) for this step.
            dt: Timestep size in years.
            t: Current simulation time.

        Returns:
            Updated ``IssuerState``.
        """
        new_cum = state.cum_intensity + intensity_increment * dt
        new_surv = jnp.exp(-new_cum)

        # Check for new default (survival dropped below threshold)
        newly_defaulted = jnp.where(
            state.has_defaulted,
            state.has_defaulted,
            jnp.where(new_surv < state.uniform_threshold, 1.0, 0.0),
        )

        # Record default time only on first default
        new_default_time = jnp.where(
            (newly_defaulted > 0.5) & (state.has_defaulted < 0.5),
            t,
            state.default_time,
        )

        return IssuerState(
            cum_intensity=new_cum,
            survival_prob=new_surv,
            uniform_threshold=state.uniform_threshold,
            has_defaulted=newly_defaulted,
            default_time=new_default_time,
        )
