"""Stochastic Volatility Jump Diffusion (SVJD) composition.

Composes existing building blocks (CIR vol, Poisson-lognormal jumps, GBM)
into the full SVJD equity/FX model matching the C# ESG engine.

The key equation:
    dS/S = (r - q - λκ)dt + σ(t)·dW + J·dN(λdt)

where:
    σ(t) comes from CIR vol process (jump-adjusted)
    J is lognormal jump size
    N is Poisson process
    κ = E[exp(J) - 1] is the jump compensator

Composition over inheritance: SVJD is assembled from protocols.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from hyesg.core.types import FXState, JumpState, ShockConfig, VolState
from hyesg.math.cir_formulas import cir_euler_step
from hyesg.math.jump_utils import jump_adjusted_sigma
from hyesg.models.equity._helpers import extract_short_rate
from hyesg.outputs import OutputName


# ── Protocols ────────────────────────────────────────────────────────────


@runtime_checkable
class VolatilityProcess(Protocol):
    """Protocol for stochastic volatility models used in SVJD composition.

    Implementations evolve variance via a CIR process and expose
    the current instantaneous volatility for the equity diffusion term.
    """

    def step_vol(
        self,
        state: VolState,
        t: float,
        dt: float,
        dW: Any,
    ) -> tuple[VolState, float]:
        """Evolve variance by one timestep and return new state + volatility.

        Args:
            state: Current variance state.
            t: Current time in years.
            dt: Timestep size in years.
            dW: Normal shock for the volatility process.

        Returns:
            Tuple of (new_vol_state, current_volatility).
        """
        ...

    def current_vol(self, state: VolState) -> Any:
        """Extract current instantaneous volatility from state.

        Args:
            state: Current variance state.

        Returns:
            Current volatility (sqrt of floored variance).
        """
        ...


@runtime_checkable
class JumpProcess(Protocol):
    """Protocol for jump models used in SVJD composition.

    Implementations sample compound Poisson jumps and provide the
    risk-neutral drift compensation term.
    """

    def sample_jumps(
        self,
        state: JumpState,
        dt: float,
        uniform: Any,
        normal: Any,
    ) -> tuple[JumpState, float, float]:
        """Sample compound Poisson jumps for this timestep.

        Args:
            state: Current jump state.
            dt: Timestep size in years.
            uniform: Uniform(0,1) shock for Poisson count.
            normal: N(0,1) shock for jump size.

        Returns:
            Tuple of (new_jump_state, log_jump_size, drift_adjustment).
        """
        ...


# ── Adapters: wrap existing models to satisfy protocols ──────────────────


class CIRVolAdapter:
    """Adapts the CIRVolatility model to the VolatilityProcess protocol.

    Wraps the existing CIR vol model step function in a functional
    interface suitable for SVJD composition.

    Args:
        alpha: Mean reversion speed.
        sigma: Vol-of-vol (volatility of variance).
        mu: Long-run mean variance.
    """

    def __init__(self, alpha: float, sigma: float, mu: float) -> None:
        self._alpha = alpha
        self._sigma = sigma
        self._mu = mu

    def step_vol(
        self,
        state: VolState,
        t: float,
        dt: float,
        dW: Any,
    ) -> tuple[VolState, Any]:
        """Advance CIR variance and return new state + volatility.

        Uses Euler-Maruyama with floored diffusion:
            V_{t+dt} = V_t + α(μ - V_t)dt + σ√max(0, V_t)·√dt·dZ

        Args:
            state: Current VolState with variance and volatility.
            t: Current time (unused for constant mu).
            dt: Timestep size.
            dW: Normal shock for variance process.

        Returns:
            (new_state, current_volatility) where volatility is from
            the START of this step (used for equity diffusion).
        """
        v = state.variance

        # Use volatility at start of step for the equity diffusion
        current_sigma = state.volatility

        # CIR Euler step with floored diffusion
        v_new, v_floor_new = cir_euler_step(v, self._alpha, self._mu, self._sigma, dt, dW)
        vol_new = jnp.sqrt(v_floor_new)

        new_state = VolState(variance=v_new, volatility=vol_new)
        return new_state, current_sigma

    def current_vol(self, state: VolState) -> Any:
        """Extract current volatility from state.

        Args:
            state: Current VolState.

        Returns:
            Current volatility (sqrt of floored variance).
        """
        return state.volatility


class ConstantJumpAdapter:
    """Adapts constant-intensity jump model to the JumpProcess protocol.

    Wraps the Merton jump process logic in a functional interface
    for SVJD composition.

    Args:
        intensity: Poisson intensity λ (jumps per year).
        jump_mean: Log-space jump mean μ_J.
        jump_vol: Log-space jump volatility σ_J.
        max_k: Maximum Poisson count for inverse CDF.
    """

    def __init__(
        self,
        intensity: float,
        jump_mean: float,
        jump_vol: float,
        max_k: int = 20,
    ) -> None:
        self._intensity = intensity
        self._jump_mean = jump_mean
        self._jump_vol = jump_vol
        self._max_k = max_k

    @property
    def intensity(self) -> float:
        """Jump intensity λ."""
        return self._intensity

    @property
    def jump_mean(self) -> float:
        """Log-space jump mean μ_J."""
        return self._jump_mean

    @property
    def jump_vol(self) -> float:
        """Log-space jump volatility σ_J."""
        return self._jump_vol

    def sample_jumps(
        self,
        state: JumpState,
        dt: float,
        uniform: Any,
        normal: Any,
    ) -> tuple[JumpState, Any, Any]:
        """Sample compound Poisson jumps for this timestep.

        Implements:
            N ~ Poisson(λ·dt) via inverse CDF
            J = N·μ_J + √N·σ_J·ε (total log-jump)
            drift_adj = -λ·(exp(μ_J + σ_J²/2) - 1)·dt

        Args:
            state: Current JumpState.
            dt: Timestep in years.
            uniform: Uniform(0,1) shock for Poisson sampling.
            normal: N(0,1) shock for jump magnitude.

        Returns:
            (new_state, log_jump_size, drift_adjustment_for_dt).
        """
        lam_dt = self._intensity * dt

        # Poisson count via inverse CDF (JIT-safe loop-unrolled version)
        n_jumps = self._poisson_inverse_cdf(uniform, lam_dt)

        # Total log-jump: N·μ + √N·σ·ε
        sqrt_n = jnp.sqrt(jnp.maximum(n_jumps, 0.0))
        jump_size = n_jumps * self._jump_mean + sqrt_n * self._jump_vol * normal
        jump_size = jnp.where(n_jumps > 0, jump_size, 0.0)

        # Risk-neutral drift compensation: -λ·κ·dt where κ = E[e^J - 1]
        kappa = jnp.exp(self._jump_mean + 0.5 * self._jump_vol**2) - 1.0
        drift_adj = -self._intensity * kappa * dt

        new_state = JumpState(
            cum_jumps=state.cum_jumps + jump_size,
            n_jumps=state.n_jumps + n_jumps,
            last_jump_size=jump_size,
        )
        return new_state, jump_size, drift_adj

    def _poisson_inverse_cdf(self, uniform: Any, lam_dt: float) -> Any:
        """Poisson inverse CDF, unrolled for JIT safety.

        Args:
            uniform: Uniform(0,1) sample.
            lam_dt: Rate parameter λ·dt.

        Returns:
            Integer Poisson count (as float for JAX compatibility).
        """
        log_pmf = -lam_dt
        cdf = jnp.exp(log_pmf)
        result = jnp.where(uniform <= cdf, 0.0, float(self._max_k))

        for k in range(1, self._max_k + 1):
            log_pmf = log_pmf + jnp.log(lam_dt + 1e-30) - jnp.log(float(k))
            cdf = cdf + jnp.exp(log_pmf)
            result = jnp.where(
                (uniform <= cdf) & (result == float(self._max_k)),
                float(k),
                result,
            )
        return result


class ZeroJumpAdapter:
    """Null-object jump adapter: no jumps ever occur.

    Satisfies the JumpProcess protocol with zero contribution,
    used when SVJD is configured without jumps.
    """

    @property
    def intensity(self) -> float:
        """Zero intensity."""
        return 0.0

    @property
    def jump_mean(self) -> float:
        """Zero jump mean."""
        return 0.0

    @property
    def jump_vol(self) -> float:
        """Zero jump vol."""
        return 0.0

    def sample_jumps(
        self,
        state: JumpState,
        dt: float,
        uniform: Any,
        normal: Any,
    ) -> tuple[JumpState, Any, Any]:
        """No-op: returns zero jump contribution.

        Args:
            state: Current JumpState (unchanged).
            dt: Timestep (unused).
            uniform: Uniform shock (unused).
            normal: Normal shock (unused).

        Returns:
            (unchanged_state, 0.0, 0.0).
        """
        zero = jnp.array(0.0, dtype=jnp.float64)
        return state, zero, zero


# ── SVJD Composition Function ────────────────────────────────────────────


def svjd_equity_step(
    log_price: Any,
    vol_state: VolState,
    jump_state: JumpState,
    vol_process: VolatilityProcess,
    jump_process: JumpProcess,
    drift: float,
    dt: float,
    dW_price: Any,
    dW_vol: Any,
    uniform: Any,
    normal_jump: Any,
) -> tuple[Any, VolState, JumpState, dict[str, Any]]:
    """Compose one SVJD equity step from vol + jump + GBM components.

    Implements the full SVJD equation in log-space:
        log S_{t+dt} = log S_t
                     + (drift - ½σ²(t))·dt          [diffusion drift]
                     + drift_adj                     [jump compensator]
                     + σ(t)·√dt·dW_price             [diffusion]
                     + J                              [jump]

    where σ(t) comes from the CIR vol process (already jump-adjusted
    at initialisation), and J is the compound Poisson log-jump.

    Args:
        log_price: Current log-price (scalar).
        vol_state: Current state of the volatility process.
        jump_state: Current state of the jump process.
        vol_process: VolatilityProcess adapter (e.g., CIRVolAdapter).
        jump_process: JumpProcess adapter (e.g., ConstantJumpAdapter).
        drift: Net drift rate (r - q for equity, r_d - r_f for FX,
               plus any market price of risk).
        dt: Timestep in years.
        dW_price: N(0,1) shock for the equity/FX price.
        dW_vol: N(0,1) shock for the volatility process.
        uniform: Uniform(0,1) shock for Poisson jump count.
        normal_jump: N(0,1) shock for jump magnitude.

    Returns:
        Tuple of (new_log_price, new_vol_state, new_jump_state, outputs).
    """
    # Step 1: Evolve volatility → get σ(t) for this step
    new_vol_state, sigma_t = vol_process.step_vol(vol_state, 0.0, dt, dW_vol)

    # Step 2: Sample jumps → get log-jump and drift compensation
    new_jump_state, jump_size, drift_adj = jump_process.sample_jumps(
        jump_state, dt, uniform, normal_jump
    )

    # Step 3: Log-normal Euler step with stochastic vol and jumps
    log_new = (
        log_price
        + (drift - 0.5 * sigma_t**2) * dt  # diffusion drift
        + drift_adj  # jump compensator
        + sigma_t * jnp.sqrt(dt) * dW_price  # diffusion
        + jump_size  # jump
    )

    outputs = {
        OutputName.LOG_RETURN: log_new - log_price,
        OutputName.SIGMA: sigma_t,
        OutputName.JUMP: jump_size,
        OutputName.DRIFT_ADJUSTMENT: drift_adj,
    }

    return log_new, new_vol_state, new_jump_state, outputs


# ── SVJDEquity Model Class ───────────────────────────────────────────────


class SVJDEquityState(NamedTuple):
    """Composite state for the SVJD equity model.

    Bundles FXState, VolState, and JumpState into a single pytree-
    compatible container for use in jax.lax.scan.

    Attributes:
        price_state: Log-price and level.
        vol_state: CIR variance state.
        jump_state: Jump accumulator state.
    """

    price_state: FXState
    vol_state: VolState
    jump_state: JumpState


class SVJDEquity:
    """Full SVJD equity model: CIR stochastic vol + Poisson-lognormal jumps.

    Composes a CIR volatility process, a constant-intensity jump model,
    and GBM into the SVJD equation. The initial sigma is jump-adjusted
    (removes jump variance from total observed volatility).

    Follows the same interface contract as the plain Equity model but
    with additional shocks for vol and jumps.

    Args:
        initial_value: Starting equity price.
        sigma: Total observed volatility (will be jump-adjusted).
        dividend_yield: Continuous dividend yield.
        vol_alpha: CIR vol mean-reversion speed.
        vol_sigma: CIR vol-of-vol.
        vol_mu: CIR vol long-run mean variance.
        jump_intensity: Poisson jump intensity λ.
        jump_mean: Log-space jump mean μ_J.
        jump_vol: Log-space jump volatility σ_J.
        name: Unique model name.
    """

    def __init__(
        self,
        initial_value: float,
        sigma: float,
        dividend_yield: float = 0.0,
        vol_alpha: float = 1.0,
        vol_sigma: float = 0.1,
        vol_mu: float = 0.04,
        jump_intensity: float = 0.0,
        jump_mean: float = 0.0,
        jump_vol: float = 0.0,
        name: str = "svjd_equity",
    ) -> None:
        self._initial_value = initial_value
        self._dividend_yield = dividend_yield
        self._name = name

        # Jump-adjusted initial sigma: removes jump variance from observed vol
        self._adjusted_sigma = float(
            jump_adjusted_sigma(sigma, jump_intensity, jump_mean, jump_vol)
        )

        # Initial variance for CIR vol process
        self._initial_variance = self._adjusted_sigma**2

        # Create component adapters
        self._vol_process = CIRVolAdapter(
            alpha=vol_alpha,
            sigma=vol_sigma,
            mu=vol_mu,
        )

        if jump_intensity > 0:
            self._jump_process: JumpProcess = ConstantJumpAdapter(
                intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_vol=jump_vol,
            )
        else:
            self._jump_process = ZeroJumpAdapter()

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def adjusted_sigma(self) -> float:
        """Jump-adjusted initial volatility."""
        return self._adjusted_sigma

    @property
    def n_shocks(self) -> int:
        """Number of shocks: price(1) + vol(1) + jump_uniform(1) + jump_normal(1)."""
        return 4

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=4,
            distribution="mixed",
            correlate=True,
            names=(
                f"{self._name}_price_z",
                f"{self._name}_vol_z",
                f"{self._name}_jump_u",
                f"{self._name}_jump_z",
            ),
        )

    def init_state(self, params: Any = None, market: Any = None) -> dict[str, Any]:
        """Create initial composite state.

        Args:
            params: Optional parameters (unused).
            market: Optional market data (unused).

        Returns:
            Dict with price_state, vol_state, jump_state keys.
        """
        s0 = jnp.array(self._initial_value, dtype=jnp.float64)
        v0 = jnp.array(self._initial_variance, dtype=jnp.float64)
        zero = jnp.array(0.0, dtype=jnp.float64)

        return {
            "price_state": FXState(log_level=jnp.log(s0), level=s0),
            "vol_state": VolState(
                variance=v0, volatility=jnp.sqrt(jnp.maximum(v0, 0.0))
            ),
            "jump_state": JumpState(
                cum_jumps=zero, n_jumps=zero, last_jump_size=zero
            ),
        }

    def step(
        self,
        state: dict[str, Any],
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Advance the SVJD equity by one timestep.

        Args:
            state: Composite state dict with price_state, vol_state, jump_state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (4,):
                [0] = N(0,1) price shock
                [1] = N(0,1) vol shock
                [2] = U(0,1) jump count shock
                [3] = N(0,1) jump size shock
            deps: Dependency outputs (expects "short_rate" from rate model).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        price_state = state["price_state"]
        vol_state = state["vol_state"]
        jump_state = state["jump_state"]

        dW_price = shocks[0]
        dW_vol = shocks[1]
        uniform = shocks[2]
        normal_jump = shocks[3]

        # Extract short rate from deps
        r = extract_short_rate(deps)

        q = jnp.array(self._dividend_yield, dtype=jnp.float64)
        drift = r - q

        # Compose SVJD step
        new_log, new_vol_state, new_jump_state, step_outputs = svjd_equity_step(
            log_price=price_state.log_level,
            vol_state=vol_state,
            jump_state=jump_state,
            vol_process=self._vol_process,
            jump_process=self._jump_process,
            drift=drift,
            dt=dt,
            dW_price=dW_price,
            dW_vol=dW_vol,
            uniform=uniform,
            normal_jump=normal_jump,
        )

        level = jnp.exp(new_log)
        new_price_state = FXState(log_level=new_log, level=level)

        new_state = {
            "price_state": new_price_state,
            "vol_state": new_vol_state,
            "jump_state": new_jump_state,
        }

        outputs = {
            OutputName.TOTAL_RETURN_INDEX: level,
            OutputName.LOG_RETURN: step_outputs[OutputName.LOG_RETURN],
            OutputName.SIGMA: step_outputs[OutputName.SIGMA],
            OutputName.JUMP: step_outputs[OutputName.JUMP],
            OutputName.DRIFT_ADJUSTMENT: step_outputs[OutputName.DRIFT_ADJUSTMENT],
        }

        return new_state, outputs
