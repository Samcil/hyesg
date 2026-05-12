"""Tests for hyesg.core.protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.protocols import (
    CreditModel,
    CurrencyAnalogy,
    ExchangeRateModel,
    Model,
    Named,
    PostProcessor,
    ShortRateModel,
    StochasticProcess,
    YieldCurveProtocol,
)
from hyesg.core.types import ShockConfig

if TYPE_CHECKING:
    from jaxtyping import Array, Float


class TestProtocolsAreRuntimeCheckable:
    """Verify all protocols have @runtime_checkable."""

    def test_named(self) -> None:
        assert hasattr(Named, "__protocol_attrs__") or hasattr(
            Named, "_is_runtime_protocol"
        )

    def test_model(self) -> None:
        assert hasattr(Model, "_is_runtime_protocol")

    def test_short_rate_model(self) -> None:
        assert hasattr(ShortRateModel, "_is_runtime_protocol")

    def test_currency_analogy(self) -> None:
        assert hasattr(CurrencyAnalogy, "_is_runtime_protocol")

    def test_exchange_rate_model(self) -> None:
        assert hasattr(ExchangeRateModel, "_is_runtime_protocol")

    def test_credit_model(self) -> None:
        assert hasattr(CreditModel, "_is_runtime_protocol")

    def test_stochastic_process(self) -> None:
        assert hasattr(StochasticProcess, "_is_runtime_protocol")

    def test_post_processor(self) -> None:
        assert hasattr(PostProcessor, "_is_runtime_protocol")

    def test_yield_curve_protocol(self) -> None:
        assert hasattr(YieldCurveProtocol, "_is_runtime_protocol")


class _DummyNamed:
    @property
    def name(self) -> str:
        return "dummy"


class _DummyModel:
    @property
    def name(self) -> str:
        return "dummy_model"

    def init_state(self, params: Any, market: Any) -> Any:
        return None

    def step(
        self,
        state: Any,
        t: float,
        dt: float,
        shocks: Float[Array, n_shocks],
        deps: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        return state, {}

    @property
    def n_shocks(self) -> int:
        return 1

    @property
    def shock_config(self) -> ShockConfig:
        return ShockConfig(
            n_shocks=1,
            distribution="normal",
            correlate=True,
            names=("z1",),
        )


class _DummyPostProcessor:
    @property
    def name(self) -> str:
        return "dummy_pp"

    def process(self, outputs: Any) -> Any:
        return outputs


class _DummyYieldCurve:
    def forward_rate(self, t: float) -> float:
        return 0.03

    def spot_rate(self, t: float) -> float:
        return 0.03

    def zcb_price(self, t: float) -> float:
        return 0.97

    def inverse_zcb_price(self, t: float) -> float:
        return 1.0 / 0.97


class _DummyStochasticProcess:
    def euler_step(
        self,
        x: Float[Array, ""],  # noqa: F722
        t: float,
        dt: float,
        dz: Float[Array, ""],  # noqa: F722
        params: Any,
    ) -> Float[Array, ""]:  # noqa: F722
        return x

    def analytic_a(self, tau: float, params: Any) -> Float[Array, ""]:  # noqa: F722
        return jnp.array(1.0)

    def analytic_b(self, tau: float, params: Any) -> Float[Array, ""]:  # noqa: F722
        return jnp.array(0.0)


class TestIsInstanceChecks:
    """Test isinstance with dummy implementations."""

    def test_named_isinstance(self) -> None:
        assert isinstance(_DummyNamed(), Named)

    def test_model_isinstance(self) -> None:
        assert isinstance(_DummyModel(), Model)

    def test_post_processor_isinstance(self) -> None:
        assert isinstance(_DummyPostProcessor(), PostProcessor)

    def test_yield_curve_isinstance(self) -> None:
        assert isinstance(_DummyYieldCurve(), YieldCurveProtocol)

    def test_stochastic_process_isinstance(self) -> None:
        assert isinstance(_DummyStochasticProcess(), StochasticProcess)

    def test_non_implementing_not_isinstance(self) -> None:
        """Plain object should not satisfy Named."""
        assert not isinstance(object(), Named)
        assert not isinstance(object(), Model)

    def test_partial_implementation_not_model(self) -> None:
        """Object with only name property is not a Model."""

        class _OnlyName:
            @property
            def name(self) -> str:
                return "x"

        obj = _OnlyName()
        assert isinstance(obj, Named)
        assert not isinstance(obj, Model)
