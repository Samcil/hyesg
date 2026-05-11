"""Core protocol definitions for hyesg models.

Every model in the system implements one or more Protocols.
These use Python's structural typing (Protocol + @runtime_checkable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from hyesg.core.types import ShockConfig


@runtime_checkable
class Named(Protocol):
    """Every component has a unique name."""

    @property
    def name(self) -> str: ...


@runtime_checkable
class Model(Protocol):
    """Core model contract. Every financial model implements this.

    The step() function is the fundamental building block:
    - Pure function (no side effects)
    - Takes current state + inputs, returns new state + outputs
    - JIT-compilable via JAX
    """

    @property
    def name(self) -> str: ...

    def init_state(self, params: Any, market: Any) -> Any: ...

    def step(
        self,
        state: Any,
        t: float,
        dt: float,
        shocks: Float[Array, "n_shocks"],
        deps: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]: ...

    @property
    def n_shocks(self) -> int: ...

    @property
    def shock_config(self) -> ShockConfig: ...


@runtime_checkable
class ShortRateModel(Model, Protocol):
    """Short rate model with analytic bond pricing."""

    def short_rate(self, state: Any) -> Float[Array, ""]: ...
    def zcb_price(self, state: Any, t: float, T: float) -> Float[Array, ""]: ...
    def spot_rate(self, state: Any, t: float, T: float) -> Float[Array, ""]: ...
    def forward_rate(self, state: Any, t: float, T: float) -> Float[Array, ""]: ...
    def swap_rate(
        self, state: Any, t: float, tenor: float, freq: float
    ) -> Float[Array, ""]: ...


@runtime_checkable
class BondOptionPricing(Protocol):
    """Analytic bond option pricing."""

    def zcb_call(
        self,
        state: Any,
        t: float,
        T: float,
        S: float,
        K: float,
    ) -> Float[Array, ""]: ...

    def zcb_put(
        self,
        state: Any,
        t: float,
        T: float,
        S: float,
        K: float,
    ) -> Float[Array, ""]: ...


@runtime_checkable
class SwaptionPricing(Protocol):
    """Swaption pricing."""

    def swaption_price(
        self,
        state: Any,
        t: float,
        T: float,
        tenor: float,
        K: float,
        freq: float,
        is_payer: bool,
    ) -> Float[Array, ""]: ...


@runtime_checkable
class CurrencyAnalogy(Model, Protocol):
    """Foreign Currency Analogy model.

    Treats real rates, inflation, dividends as 'currencies' with
    exchange rates, enabling unified pricing via ZCB prices.
    """

    def exchange_rate(self, state: Any, t: float) -> Float[Array, ""]: ...
    def zcb_price(self, state: Any, t: float, T: float) -> Float[Array, ""]: ...
    def spot_rate(self, state: Any, t: float, T: float) -> Float[Array, ""]: ...


@runtime_checkable
class ExchangeRateModel(Model, Protocol):
    """FX / inflation exchange rate."""

    def level(self, state: Any, t: float) -> Float[Array, ""]: ...
    def log_return(self, state: Any, t: float, dt: float) -> Float[Array, ""]: ...


@runtime_checkable
class CreditModel(Model, Protocol):
    """Credit default intensity model."""

    def default_intensity(self, state: Any, t: float) -> Float[Array, ""]: ...
    def survival_probability(
        self, state: Any, t: float, T: float
    ) -> Float[Array, ""]: ...
    def has_defaulted(self, state: Any) -> Float[Array, ""]: ...


@runtime_checkable
class PortfolioModel(Model, Protocol):
    """Portfolio with rebalancing."""

    def value(self, state: Any, t: float) -> Float[Array, ""]: ...
    def income(self, state: Any, t: float, dt: float) -> Float[Array, ""]: ...


@runtime_checkable
class StochasticProcess(Protocol):
    """Low-level SDE discretisation.

    Models delegate to euler_step() for the actual SDE advancement.
    analytic_a() and analytic_b() are used for ZCB pricing in
    CIR and G++ models:
    P(t,T) = A(tau) * exp(-B(tau) * x(t)) where tau = T - t.
    """

    def euler_step(
        self,
        x: Float[Array, ""],
        t: float,
        dt: float,
        dz: Float[Array, ""],
        params: Any,
    ) -> Float[Array, ""]: ...

    def analytic_a(self, tau: float, params: Any) -> Float[Array, ""]: ...

    def analytic_b(self, tau: float, params: Any) -> Float[Array, ""]: ...


@runtime_checkable
class PostProcessor(Protocol):
    """Post-simulation processing."""

    @property
    def name(self) -> str: ...

    def process(self, outputs: Any) -> Any: ...


@runtime_checkable
class YieldCurveProtocol(Protocol):
    """Yield curve interface."""

    def forward_rate(self, t: float) -> float: ...
    def spot_rate(self, t: float) -> float: ...
    def zcb_price(self, t: float) -> float: ...
    def inverse_zcb_price(self, t: float) -> float: ...
