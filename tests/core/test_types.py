"""Tests for hyesg.core.types."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.core.types import (
    CIR2State,
    CIRState,
    CreditState,
    FXState,
    G2State,
    OUState,
    OutputSpec,
    PortfolioState,
    ShockConfig,
    SimulationState,
    TimeStep,
)


class TestTimeStep:
    def test_creation(self) -> None:
        ts = TimeStep(index=0, time=0.0, dt=1 / 12, is_zero=True)
        assert ts.index == 0
        assert ts.time == 0.0
        assert ts.is_zero is True

    def test_immutability(self) -> None:
        ts = TimeStep(index=0, time=0.0, dt=1 / 12, is_zero=True)
        with pytest.raises(AttributeError):
            ts.index = 1  # type: ignore[misc]


class TestShockConfig:
    def test_creation(self) -> None:
        sc = ShockConfig(
            n_shocks=2,
            distribution="normal",
            correlate=True,
            names=("z1", "z2"),
        )
        assert sc.n_shocks == 2
        assert sc.names == ("z1", "z2")

    def test_immutability(self) -> None:
        sc = ShockConfig(
            n_shocks=1,
            distribution="normal",
            correlate=True,
            names=("z",),
        )
        with pytest.raises(AttributeError):
            sc.n_shocks = 5  # type: ignore[misc]


class TestCIRState:
    def test_creation(self) -> None:
        s = CIRState(
            x=jnp.array(0.03),
            state_var=jnp.array(0.03),
            short_rate=jnp.array(0.04),
        )
        assert float(s.x) == pytest.approx(0.03)
        assert float(s.state_var) == pytest.approx(0.03)
        assert float(s.short_rate) == pytest.approx(0.04)

    def test_jax_pytree_leaves(self) -> None:
        s = CIRState(
            x=jnp.array(0.01),
            state_var=jnp.array(0.01),
            short_rate=jnp.array(0.02),
        )
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == 3
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_jax_tree_map(self) -> None:
        s = CIRState(
            x=jnp.array(1.0),
            state_var=jnp.array(2.0),
            short_rate=jnp.array(3.0),
        )
        doubled = jax.tree_util.tree_map(lambda x: x * 2.0, s)
        assert float(doubled.x) == pytest.approx(2.0)
        assert float(doubled.state_var) == pytest.approx(4.0)
        assert float(doubled.short_rate) == pytest.approx(6.0)


class TestCIR2State:
    def test_creation(self) -> None:
        s = CIR2State(
            x1=jnp.array(0.01),
            x2=jnp.array(0.02),
            state_var1=jnp.array(0.01),
            state_var2=jnp.array(0.02),
            short_rate=jnp.array(0.05),
        )
        assert float(s.short_rate) == pytest.approx(0.05)

    def test_jax_pytree_leaves(self) -> None:
        s = CIR2State(
            x1=jnp.array(0.01),
            x2=jnp.array(0.02),
            state_var1=jnp.array(0.01),
            state_var2=jnp.array(0.02),
            short_rate=jnp.array(0.05),
        )
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == 5


class TestOUState:
    def test_creation(self) -> None:
        s = OUState(
            x=jnp.array(-0.01),
            short_rate=jnp.array(-0.005),
        )
        assert float(s.x) == pytest.approx(-0.01)

    def test_negative_values(self) -> None:
        """OU can go negative — verify no clamping."""
        s = OUState(
            x=jnp.array(-0.5),
            short_rate=jnp.array(-0.4),
        )
        assert float(s.x) < 0


class TestG2State:
    def test_creation_and_pytree(self) -> None:
        s = G2State(
            x1=jnp.array(0.01),
            x2=jnp.array(-0.01),
            short_rate=jnp.array(0.02),
        )
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == 3


class TestCreditState:
    def test_creation(self) -> None:
        s = CreditState(
            intensity=jnp.array(0.02),
            cum_intensity=jnp.array(0.1),
            has_defaulted=jnp.array(0.0),
        )
        assert float(s.has_defaulted) == 0.0


class TestFXState:
    def test_creation(self) -> None:
        s = FXState(
            log_level=jnp.array(0.0),
            level=jnp.array(1.0),
        )
        assert float(s.level) == pytest.approx(1.0)


class TestPortfolioState:
    def test_creation(self) -> None:
        s = PortfolioState(
            value=jnp.array(100.0),
            income=jnp.array(0.0),
            weights=jnp.array([0.6, 0.4]),
        )
        assert float(s.value) == pytest.approx(100.0)
        assert s.weights.shape == (2,)


class TestSimulationState:
    def test_creation(self) -> None:
        cir = CIRState(
            x=jnp.array(0.03),
            state_var=jnp.array(0.03),
            short_rate=jnp.array(0.04),
        )
        s = SimulationState(
            model_states={"nominal": cir},
            t=0.0,
            step_index=0,
        )
        assert s.step_index == 0
        assert "nominal" in s.model_states


class TestOutputSpec:
    def test_creation(self) -> None:
        spec = OutputSpec(
            model_name="nominal",
            member_name="short_rate",
            output_name="ShortRate",
        )
        assert spec.args == ()

    def test_with_args(self) -> None:
        spec = OutputSpec(
            model_name="nominal",
            member_name="spot_rate",
            output_name="20ySpotRate",
            args=(20.0,),
        )
        assert spec.args == (20.0,)
