"""Core state types for hyesg models.

All state containers are NamedTuples so they work as JAX pytrees
out of the box — no registration required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jaxtyping import Array, Float


class TimeStep(NamedTuple):
    """A single point on the simulation time grid.

    Attributes:
        index: Step index (0-based).
        time: Absolute time in years.
        dt: Size of this timestep in years.
        is_zero: Whether this is the initial (t=0) step.
    """

    index: int
    time: float
    dt: float
    is_zero: bool


class ShockConfig(NamedTuple):
    """Metadata describing the shocks a model requires.

    Attributes:
        n_shocks: Number of independent Brownian increments.
        distribution: Shock distribution type (e.g. "normal").
        correlate: Whether shocks participate in correlation.
        names: Human-readable names for each shock stream.
    """

    n_shocks: int
    distribution: str
    correlate: bool
    names: tuple[str, ...]


class CIRState(NamedTuple):
    """State of a single-factor CIR process.

    Attributes:
        x: Internal raw state variable (may go slightly negative).
        state_var: Floored state variable, max(0, x).
        short_rate: Observable short rate (x + phi or state_var).
    """

    x: Float[Array, ""]
    state_var: Float[Array, ""]
    short_rate: Float[Array, ""]


class CIR2State(NamedTuple):
    """State of a two-factor CIR++ model (nominal rates).

    Attributes:
        x1: Raw state of factor 1.
        x2: Raw state of factor 2.
        state_var1: Floored state of factor 1.
        state_var2: Floored state of factor 2.
        short_rate: Observable short rate (phi + x1 + x2).
    """

    x1: Float[Array, ""]
    x2: Float[Array, ""]
    state_var1: Float[Array, ""]
    state_var2: Float[Array, ""]
    short_rate: Float[Array, ""]


class OUState(NamedTuple):
    """State of an Ornstein-Uhlenbeck (Vasicek/G1++) process.

    Attributes:
        x: Internal state variable (can be negative).
        short_rate: Observable short rate.
    """

    x: Float[Array, ""]
    short_rate: Float[Array, ""]


class G2State(NamedTuple):
    """State of a two-factor Gaussian model (G2++, real rates).

    Attributes:
        x1: State of factor 1.
        x2: State of factor 2.
        short_rate: Observable real short rate (psi + x1 + x2).
    """

    x1: Float[Array, ""]
    x2: Float[Array, ""]
    short_rate: Float[Array, ""]


class CreditState(NamedTuple):
    """State of a credit default intensity model.

    Attributes:
        intensity: Current default intensity (lambda).
        cum_intensity: Cumulative integrated intensity.
        has_defaulted: 1.0 if defaulted, 0.0 otherwise.
    """

    intensity: Float[Array, ""]
    cum_intensity: Float[Array, ""]
    has_defaulted: Float[Array, ""]


class VolState(NamedTuple):
    """State of a stochastic volatility process.

    Attributes:
        variance: Current variance V(t).
        volatility: Current volatility √V(t).
    """

    variance: Float[Array, ""]
    volatility: Float[Array, ""]


class JumpState(NamedTuple):
    """State of a jump diffusion process.

    Attributes:
        cum_jumps: Cumulative log-jump contribution.
        n_jumps: Total number of jumps so far.
        last_jump_size: Size of last jump (for diagnostics).
    """

    cum_jumps: Float[Array, ""]
    n_jumps: Float[Array, ""]
    last_jump_size: Float[Array, ""]


class FXState(NamedTuple):
    """State of an exchange rate / inflation index model.

    Attributes:
        log_level: Log of the current level.
        level: Current level (exp of log_level).
    """

    log_level: Float[Array, ""]
    level: Float[Array, ""]


class PortfolioState(NamedTuple):
    """State of a portfolio model.

    Attributes:
        value: Current portfolio value.
        income: Accumulated income.
        weights: Current asset weights.
    """

    value: Float[Array, ""]
    income: Float[Array, ""]
    weights: Float[Array, "n"]


class SimulationState(NamedTuple):
    """Composite carry state for the simulation scan loop.

    Attributes:
        model_states: Dict mapping model name to its current state.
        t: Current simulation time.
        step_index: Current step index.
    """

    model_states: dict[str, object]
    t: float
    step_index: int


class OutputSpec(NamedTuple):
    """Specification for extracting a single output from simulation.

    Attributes:
        model_name: Name of the source model.
        member_name: State member or method to extract.
        output_name: Canonical output name for downstream use.
        args: Extra arguments (e.g. maturity for spot rate).
    """

    model_name: str
    member_name: str
    output_name: str
    args: tuple[float, ...] = ()
