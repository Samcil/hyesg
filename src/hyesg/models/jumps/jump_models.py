"""Merton jump diffusion models: zero, constant-intensity, stochastic-intensity.

Implements jump processes that can be composed with continuous diffusion
models (e.g. GBM equity).  All three variants share the ``JumpState``
carry type and produce the same output dict shape so they are
interchangeable at the orchestration layer.

Jump process (Merton):
    N(t) ~ Poisson(λ·dt)
    J = N·μ_J + √N·σ_J·ε       (total log-jump)

Risk-neutral drift adjustment:
    drift_adj = -λ·(exp(μ_J + σ_J²/2) - 1)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import JumpState, ShockConfig
from hyesg.outputs import OutputName


# ── Poisson inverse-CDF helper ──────────────────────────────────────────


def _poisson_inverse_cdf(uniform: jnp.ndarray, lam: float, max_k: int = 20) -> jnp.ndarray:
    """Inverse CDF for Poisson distribution, JIT-safe.

    Walks the CDF of Poisson(lam) at integer k = 0, 1, …, max_k and
    returns the smallest k whose cumulative probability exceeds the
    supplied uniform sample.

    Args:
        uniform: Uniform(0, 1) samples, arbitrary shape.
        lam: Poisson rate parameter (λ·dt).
        max_k: Maximum number of jumps to consider.

    Returns:
        Poisson samples with the same shape as *uniform*.
    """
    log_pmf = -lam
    cdf = jnp.exp(log_pmf)
    result = jnp.where(uniform <= cdf, 0.0, float(max_k))

    for k in range(1, max_k + 1):
        log_pmf = log_pmf + jnp.log(lam + 1e-30) - jnp.log(float(k))
        cdf = cdf + jnp.exp(log_pmf)
        result = jnp.where(
            (uniform <= cdf) & (result == float(max_k)),
            float(k),
            result,
        )

    return result


def _poisson_continuous_approx(
    uniform: jnp.ndarray, lam: float, max_k: int = 20
) -> jnp.ndarray:
    """Continuous approximation: linear interpolation of Poisson CDF.

    Builds the CDF at integer points 0 … max_k and uses
    ``jnp.interp`` to return a continuous-valued count.

    Args:
        uniform: Uniform(0, 1) samples, arbitrary shape.
        lam: Poisson rate parameter (λ·dt).
        max_k: Maximum number of jumps to consider.

    Returns:
        Continuous-valued Poisson approximation, same shape as *uniform*.
    """
    k_values = jnp.arange(max_k + 1, dtype=jnp.float64)
    log_pmf = -lam
    cumsum = jnp.exp(log_pmf)
    cdf_list = [cumsum]
    for k in range(1, max_k + 1):
        log_pmf = log_pmf + jnp.log(lam + 1e-30) - jnp.log(float(k))
        cumsum = cumsum + jnp.exp(log_pmf)
        cdf_list.append(cumsum)
    cdf_vals = jnp.stack(cdf_list)

    flat = uniform.ravel()
    interped = jnp.interp(flat, cdf_vals, k_values)
    return interped.reshape(uniform.shape)


# ── Helper for jump step logic ───────────────────────────────────────────


def _jump_step(
    state: JumpState,
    uniform: jnp.ndarray,
    normal: jnp.ndarray,
    intensity: float,
    jump_mean: float,
    jump_vol: float,
    dt: float,
    poisson_fn,
) -> tuple[JumpState, dict[str, Any]]:
    """Shared jump step logic for constant and stochastic models.

    Args:
        state: Current ``JumpState``.
        uniform: Uniform shock for Poisson count.
        normal: Normal shock for jump size.
        intensity: Current jump intensity λ.
        jump_mean: Log-space jump mean μ_J.
        jump_vol: Log-space jump volatility σ_J.
        dt: Timestep in years.
        poisson_fn: Callable(uniform, lam_dt) → count.

    Returns:
        (new_state, outputs) tuple.
    """
    lam_dt = intensity * dt

    # Number of jumps in this timestep
    n_jumps = poisson_fn(uniform, lam_dt)

    # Total jump size: N·μ + √N·σ·ε (zero when N = 0)
    sqrt_n = jnp.sqrt(jnp.maximum(n_jumps, 0.0))
    jump_size = n_jumps * jump_mean + sqrt_n * jump_vol * normal
    jump_size = jnp.where(n_jumps > 0, jump_size, 0.0)

    # Risk-neutral drift adjustment
    drift_adj = -intensity * (jnp.exp(jump_mean + 0.5 * jump_vol**2) - 1.0)

    new_state = JumpState(
        cum_jumps=state.cum_jumps + jump_size,
        n_jumps=state.n_jumps + n_jumps,
        last_jump_size=jump_size,
    )
    outputs = {
        OutputName.JUMP: jump_size,
        OutputName.DRIFT_ADJUSTMENT: drift_adj * dt,
        OutputName.N_JUMPS: n_jumps,
    }
    return new_state, outputs


# ── ZeroJumpModel ────────────────────────────────────────────────────────


@register_model("zero_jump")
class ZeroJumpModel:
    """Null-object jump model — no jumps ever occur.

    Used as the default when jump diffusion is not needed.
    Implements the same interface as other jump models so it can
    be swapped in without changing caller code.

    Args:
        name: Unique model name.
    """

    def __init__(self, name: str = "zero_jump") -> None:
        self._name = name

    # ─── Properties ───

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def n_shocks(self) -> int:
        """Number of independent shocks (zero — no randomness)."""
        return 0

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=0,
            distribution="none",
            correlate=False,
            names=(),
        )

    # ─── State management ───

    def init_state(self, params: Any = None, market: Any = None) -> JumpState:
        """Create initial (zero) jump state.

        Args:
            params: Unused.
            market: Unused.

        Returns:
            ``JumpState`` with all fields zeroed.
        """
        zero = jnp.array(0.0, dtype=jnp.float64)
        return JumpState(cum_jumps=zero, n_jumps=zero, last_jump_size=zero)

    # ─── Step function ───

    def step(
        self,
        state: JumpState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[JumpState, dict[str, Any]]:
        """No-op step: returns zero jump contribution.

        Args:
            state: Current ``JumpState``.
            t: Current time in years.
            dt: Timestep size.
            shocks: Ignored (model requires 0 shocks).
            deps: Dependency outputs (unused).

        Returns:
            (unchanged state, zero-valued outputs).
        """
        zero = jnp.array(0.0, dtype=jnp.float64)
        outputs = {
            OutputName.JUMP: zero,
            OutputName.DRIFT_ADJUSTMENT: zero,
            OutputName.N_JUMPS: zero,
        }
        return state, outputs


# ── ConstantIntensityJumpModel ───────────────────────────────────────────


@register_model("constant_jump")
class ConstantIntensityJumpModel:
    """Merton jump diffusion with constant intensity λ.

    Jump process::

        N(t) ~ Poisson(λ·dt)        — number of jumps in [t, t+dt]
        J = N·μ_J + √N·σ_J·ε        — total log-jump size

    where ε ~ N(0, 1).

    Risk-neutral drift adjustment::

        drift_adj = -λ·(exp(μ_J + σ_J²/2) - 1)

    This model requires **2 shocks**:

    * ``shocks[0]``: uniform for Poisson count (inverse CDF).
    * ``shocks[1]``: normal for jump size.

    Args:
        intensity: Poisson intensity λ.
        jump_mean: Log-space jump mean μ_J.
        jump_vol: Log-space jump volatility σ_J.
        poisson_type: ``"exact"`` (discrete CDF) or ``"continuous"``
            (linear interpolation of CDF).
        name: Unique model name.
    """

    def __init__(
        self,
        intensity: float,
        jump_mean: float,
        jump_vol: float,
        poisson_type: str = "exact",
        name: str = "constant_jump",
    ) -> None:
        self._intensity = intensity
        self._jump_mean = jump_mean
        self._jump_vol = jump_vol
        self._poisson_type = poisson_type
        self._name = name

    # ─── Properties ───

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def n_shocks(self) -> int:
        """Two shocks: uniform (Poisson) + normal (size)."""
        return 2

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=2,
            distribution="mixed",
            correlate=False,
            names=(f"{self._name}_uniform", f"{self._name}_normal"),
        )

    # ─── State management ───

    def init_state(self, params: Any = None, market: Any = None) -> JumpState:
        """Create initial (zero) jump state.

        Args:
            params: Unused.
            market: Unused.

        Returns:
            ``JumpState`` with all fields zeroed.
        """
        zero = jnp.array(0.0, dtype=jnp.float64)
        return JumpState(cum_jumps=zero, n_jumps=zero, last_jump_size=zero)

    # ─── Poisson sampling ───

    def _sample_poisson(
        self, uniform: jnp.ndarray, lam_dt: float
    ) -> jnp.ndarray:
        """Sample Poisson count using the configured method.

        Args:
            uniform: Uniform(0, 1) sample.
            lam_dt: Rate parameter λ·dt.

        Returns:
            Integer-valued Poisson count (as float for JAX).
        """
        if self._poisson_type == "continuous":
            return _poisson_continuous_approx(uniform, lam_dt)
        return _poisson_inverse_cdf(uniform, lam_dt)

    # ─── Step function ───

    def step(
        self,
        state: JumpState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[JumpState, dict[str, Any]]:
        """Advance the jump process by one timestep.

        Args:
            state: Current ``JumpState``.
            t: Current time in years.
            dt: Timestep size.
            shocks: Array of shape ``(2,)`` — ``[uniform, normal]``.
            deps: Dependency outputs (unused).

        Returns:
            (new_state, outputs) tuple.
        """
        return _jump_step(
            state=state,
            uniform=shocks[0],
            normal=shocks[1],
            intensity=self._intensity,
            jump_mean=self._jump_mean,
            jump_vol=self._jump_vol,
            dt=dt,
            poisson_fn=self._sample_poisson,
        )


# ── StochasticIntensityJumpModel ─────────────────────────────────────────


@register_model("stochastic_jump")
class StochasticIntensityJumpModel:
    """Merton jump diffusion with CIR-driven stochastic intensity.

    The jump intensity λ(t) is read from a dependency (typically a CIR
    volatility model)::

        λ(t) = deps[intensity_dep]["variance"]

    Otherwise identical to :class:`ConstantIntensityJumpModel`.

    Args:
        jump_mean: Log-space jump mean μ_J.
        jump_vol: Log-space jump volatility σ_J.
        intensity_dep: Name of the dependency providing ``"variance"``.
        poisson_type: ``"exact"`` or ``"continuous"``.
        name: Unique model name.
    """

    def __init__(
        self,
        jump_mean: float,
        jump_vol: float,
        intensity_dep: str = "cir_vol",
        poisson_type: str = "exact",
        name: str = "stochastic_jump",
    ) -> None:
        self._jump_mean = jump_mean
        self._jump_vol = jump_vol
        self._intensity_dep = intensity_dep
        self._poisson_type = poisson_type
        self._name = name

    # ─── Properties ───

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def n_shocks(self) -> int:
        """Two shocks: uniform (Poisson) + normal (size)."""
        return 2

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=2,
            distribution="mixed",
            correlate=False,
            names=(f"{self._name}_uniform", f"{self._name}_normal"),
        )

    # ─── State management ───

    def init_state(self, params: Any = None, market: Any = None) -> JumpState:
        """Create initial (zero) jump state.

        Args:
            params: Unused.
            market: Unused.

        Returns:
            ``JumpState`` with all fields zeroed.
        """
        zero = jnp.array(0.0, dtype=jnp.float64)
        return JumpState(cum_jumps=zero, n_jumps=zero, last_jump_size=zero)

    # ─── Poisson sampling ───

    def _sample_poisson(
        self, uniform: jnp.ndarray, lam_dt: float
    ) -> jnp.ndarray:
        """Sample Poisson count using the configured method.

        Args:
            uniform: Uniform(0, 1) sample.
            lam_dt: Rate parameter λ·dt.

        Returns:
            Integer-valued Poisson count (as float for JAX).
        """
        if self._poisson_type == "continuous":
            return _poisson_continuous_approx(uniform, lam_dt)
        return _poisson_inverse_cdf(uniform, lam_dt)

    # ─── Step function ───

    def step(
        self,
        state: JumpState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[JumpState, dict[str, Any]]:
        """Advance the jump process by one timestep.

        Reads intensity from the dependency specified at construction.

        Args:
            state: Current ``JumpState``.
            t: Current time in years.
            dt: Timestep size.
            shocks: Array of shape ``(2,)`` — ``[uniform, normal]``.
            deps: Dependency outputs; must contain
                ``deps[intensity_dep]["variance"]``.

        Returns:
            (new_state, outputs) tuple.
        """
        intensity = deps[self._intensity_dep][OutputName.VARIANCE]
        return _jump_step(
            state=state,
            uniform=shocks[0],
            normal=shocks[1],
            intensity=intensity,
            jump_mean=self._jump_mean,
            jump_vol=self._jump_vol,
            dt=dt,
            poisson_fn=self._sample_poisson,
        )
