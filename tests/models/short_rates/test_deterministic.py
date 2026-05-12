"""Tests for the deterministic short rate model."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from hyesg.core.registry import clear_registry, get_model
from hyesg.math.curves.parametric import NelsonSiegelCurve
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.models.short_rates.deterministic import Deterministic

jax.config.update("jax_enable_x64", True)

FLAT_RATE = 0.04


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the Deterministic model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.deterministic as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def flat_curve() -> ConstantCurve:
    """A flat forward-rate curve at 4%."""
    return ConstantCurve(FLAT_RATE)


@pytest.fixture
def ns_curve() -> NelsonSiegelCurve:
    """A Nelson-Siegel curve with typical GBP-like parameters."""
    return NelsonSiegelCurve(beta0=0.04, beta1=-0.02, beta2=0.01, tau=2.0)


@pytest.fixture
def flat_model(flat_curve: ConstantCurve) -> Deterministic:
    """Deterministic model backed by a flat curve."""
    return Deterministic(flat_curve)


@pytest.fixture
def ns_model(ns_curve: NelsonSiegelCurve) -> Deterministic:
    """Deterministic model backed by a Nelson-Siegel curve."""
    return Deterministic(ns_curve, name="det_ns")


# ─── Metadata ───


class TestDeterministicInit:
    """Tests for model construction and metadata."""

    def test_name(self, flat_model: Deterministic) -> None:
        assert flat_model.name == "deterministic"

    def test_custom_name(self, ns_model: Deterministic) -> None:
        assert ns_model.name == "det_ns"

    def test_n_shocks(self, flat_model: Deterministic) -> None:
        assert flat_model.n_shocks == 0

    def test_shock_config(self, flat_model: Deterministic) -> None:
        cfg = flat_model.shock_config
        assert cfg.n_shocks == 0
        assert cfg.distribution == "normal"
        assert cfg.correlate is False
        assert cfg.names == ()

    def test_registry(self) -> None:
        """Deterministic should be retrievable from the registry."""
        cls = get_model("deterministic")
        assert cls.__name__ == "Deterministic"


# ─── init_state ───


class TestDeterministicInitState:
    """Tests for init_state."""

    def test_state_type(self, flat_model: Deterministic) -> None:
        state = flat_model.init_state()
        assert hasattr(state, "x")
        assert hasattr(state, "short_rate")

    def test_initial_rate_flat(self, flat_model: Deterministic) -> None:
        """init_state short_rate should equal f(0) = flat rate."""
        state = flat_model.init_state()
        assert float(state.short_rate) == pytest.approx(FLAT_RATE, abs=1e-12)
        assert float(state.x) == pytest.approx(0.0, abs=1e-12)

    def test_initial_rate_ns(self, ns_model: Deterministic) -> None:
        """init_state short_rate should equal NS curve at t=0."""
        state = ns_model.init_state()
        expected = 0.04 + (-0.02)  # beta0 + beta1 at t=0
        assert float(state.short_rate) == pytest.approx(expected, abs=1e-10)


# ─── step ───


class TestDeterministicStep:
    """Tests for the step function."""

    def test_step_preserves_flat_rate(self, flat_model: Deterministic) -> None:
        """With a flat curve, every step should produce the same rate."""
        state = flat_model.init_state()
        for t in [0.0, 0.25, 0.5, 1.0]:
            new_state, outputs = flat_model.step(state, t, 0.25, jnp.array([]), {})
            assert float(outputs["ShortRate"]) == pytest.approx(FLAT_RATE, abs=1e-12)
            state = new_state

    def test_step_reads_ns_curve(
        self, ns_model: Deterministic, ns_curve: NelsonSiegelCurve
    ) -> None:
        """step() should return the forward rate from the NS curve."""
        state = ns_model.init_state()
        dt = 0.25
        for i in range(4):
            t = i * dt
            new_state, outputs = ns_model.step(state, t, dt, jnp.array([]), {})
            expected = ns_curve.evaluate(t + dt)
            assert float(outputs["ShortRate"]) == pytest.approx(expected, abs=1e-10)
            state = new_state

    def test_step_x_stays_zero(self, flat_model: Deterministic) -> None:
        """The internal state variable x should remain zero."""
        state = flat_model.init_state()
        new_state, _ = flat_model.step(state, 0.0, 0.25, jnp.array([]), {})
        assert float(new_state.x) == pytest.approx(0.0, abs=1e-12)

    def test_step_output_keys(self, flat_model: Deterministic) -> None:
        state = flat_model.init_state()
        _, outputs = flat_model.step(state, 0.0, 0.25, jnp.array([]), {})
        assert "ShortRate" in outputs


# ─── Analytics ───


class TestDeterministicAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, flat_model: Deterministic) -> None:
        state = flat_model.init_state()
        assert float(flat_model.short_rate(state)) == pytest.approx(FLAT_RATE, abs=1e-12)

    def test_zcb_at_zero_tau(self, flat_model: Deterministic) -> None:
        """P(t, t) = 1."""
        state = flat_model.init_state()
        p = flat_model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-12)

    def test_zcb_flat_curve(self, flat_model: Deterministic) -> None:
        """P(0, T) = exp(-r*T) for a flat curve."""
        state = flat_model.init_state()
        for maturity in [1.0, 5.0, 10.0]:
            p = float(flat_model.zcb_price(state, 0.0, maturity))
            expected = math.exp(-FLAT_RATE * maturity)
            assert p == pytest.approx(expected, abs=1e-8)

    def test_zcb_forward_price(self, flat_model: Deterministic) -> None:
        """P(t, T) = P(0,T) / P(0,t)."""
        state = flat_model.init_state()
        t, maturity = 2.0, 7.0
        p_t_T = float(flat_model.zcb_price(state, t, maturity))
        p_0_T = float(flat_model.zcb_price(state, 0.0, maturity))
        p_0_t = float(flat_model.zcb_price(state, 0.0, t))
        assert p_t_T == pytest.approx(p_0_T / p_0_t, abs=1e-10)

    def test_zcb_in_unit_interval(self, flat_model: Deterministic) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = flat_model.init_state()
        for maturity in [1.0, 5.0, 10.0]:
            p = float(flat_model.zcb_price(state, 0.0, maturity))
            assert 0.0 < p < 1.0

    def test_zcb_decreasing_with_maturity(self, flat_model: Deterministic) -> None:
        """Longer maturity → lower ZCB price."""
        state = flat_model.init_state()
        prices = [float(flat_model.zcb_price(state, 0.0, m)) for m in [1.0, 5.0, 10.0]]
        assert prices[0] > prices[1] > prices[2]

    def test_spot_rate_flat(self, flat_model: Deterministic) -> None:
        """Spot rate for a flat curve should equal the flat rate."""
        state = flat_model.init_state()
        r = float(flat_model.spot_rate(state, 0.0, 5.0))
        assert r == pytest.approx(FLAT_RATE, abs=1e-8)

    def test_spot_rate_consistent_with_zcb(
        self, ns_model: Deterministic
    ) -> None:
        """R(t,T) = -ln P(t,T) / (T-t)."""
        state = ns_model.init_state()
        t, maturity = 1.0, 6.0
        p = float(ns_model.zcb_price(state, t, maturity))
        r = float(ns_model.spot_rate(state, t, maturity))
        expected = -math.log(p) / (maturity - t)
        assert r == pytest.approx(expected, abs=1e-10)

    def test_forward_rate_matches_curve(
        self, ns_model: Deterministic, ns_curve: NelsonSiegelCurve
    ) -> None:
        """forward_rate should return the curve value at maturity."""
        state = ns_model.init_state()
        for maturity in [0.5, 1.0, 5.0, 10.0]:
            f = float(ns_model.forward_rate(state, 0.0, maturity))
            expected = ns_curve.evaluate(maturity)
            assert f == pytest.approx(expected, abs=1e-10)

    def test_swap_rate_positive(self, flat_model: Deterministic) -> None:
        state = flat_model.init_state()
        s = float(flat_model.swap_rate(state, 0.0, 5.0, 1.0))
        assert s > 0.0

    def test_swap_rate_flat_curve(self, flat_model: Deterministic) -> None:
        """For a flat curve the par swap rate should be close to the flat rate.

        With discrete annual payments and continuous compounding, the par
        swap rate is slightly above the flat rate.  The gap shrinks as
        payment frequency increases.
        """
        state = flat_model.init_state()
        # Semi-annual payments → very close to the flat rate
        s = float(flat_model.swap_rate(state, 0.0, 10.0, 0.5))
        assert s == pytest.approx(FLAT_RATE, abs=5e-4)
