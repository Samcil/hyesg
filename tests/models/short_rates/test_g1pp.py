"""Tests for the G1++ (Hull-White) short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import OUParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.short_rates.g1pp import G1PP

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
SIGMA = 0.01
X0 = 0.0
FLAT_RATE = 0.03
MU = 0.03


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the G1PP model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.g1pp as mod

    importlib.reload(mod)
    yield
    clear_registry()


class FlatForwardCurve(ParametricCurve):
    """Flat forward rate curve for testing."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate


@pytest.fixture
def params() -> OUParams:
    """Standard G1++ parameters (mu=0 enforced)."""
    return OUParams(
        alpha=ALPHA,
        mu=0.0,
        sigma=SIGMA,
        initial_value=X0,
        model_type="g1pp",
    )


@pytest.fixture
def market_curve() -> FlatForwardCurve:
    """Flat forward curve at 3%."""
    return FlatForwardCurve(FLAT_RATE)


@pytest.fixture
def model(params: OUParams, market_curve: FlatForwardCurve) -> G1PP:
    """Standard G1++ model instance."""
    return G1PP(params, market_curve)


class TestG1PPInit:
    """Tests for G1++ model construction and metadata."""

    def test_name(self, model: G1PP) -> None:
        assert model.name == "g1pp"

    def test_custom_name(
        self, params: OUParams, market_curve: FlatForwardCurve
    ) -> None:
        m = G1PP(params, market_curve, name="my_g1pp")
        assert m.name == "my_g1pp"

    def test_n_shocks(self, model: G1PP) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: G1PP) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True

    def test_registry(self) -> None:
        """G1PP should be retrievable from registry."""
        cls = get_model("g1pp")
        assert cls.__name__ == "G1PP"

    def test_wrong_model_type(self, market_curve: FlatForwardCurve) -> None:
        """Should reject non-g1pp model_type."""
        params = OUParams(alpha=ALPHA, mu=MU, sigma=SIGMA, model_type="vasicek")
        with pytest.raises(ValueError, match="model_type='g1pp'"):
            G1PP(params, market_curve)


class TestG1PPInitState:
    """Tests for init_state."""

    def test_initial_x(self, model: G1PP) -> None:
        state = model.init_state()
        assert float(state.x) == pytest.approx(X0, abs=1e-12)

    def test_initial_short_rate(self, model: G1PP) -> None:
        """r(0) = x(0) + φ(0), and φ(0) = f(0,0) + 0 = flat_rate."""
        state = model.init_state()
        # φ(0) = f(0,0) + (σ²/2α²)(1-e^0)² = flat_rate + 0
        assert float(state.short_rate) == pytest.approx(FLAT_RATE, abs=1e-10)


class TestG1PPStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: G1PP) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "short_rate" in outputs

    def test_zero_shock(self, model: G1PP) -> None:
        """With zero shock and x=0, x should stay near zero."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # x evolves as x - α·x·dt = 0 - 0 = 0
        assert float(new_state.x) == pytest.approx(0.0, abs=1e-10)


class TestG1PPAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: G1PP) -> None:
        state = model.init_state()
        r_val = model.short_rate(state)
        assert float(r_val) == pytest.approx(FLAT_RATE, abs=1e-10)

    def test_zcb_at_zero_tau(self, model: G1PP) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-10)

    def test_zcb_matches_market_at_t0(self, model: G1PP) -> None:
        """At t=0, x=0: P(0,T) should match the market ZCB price.

        For flat forward rate f:
            P₀(T) = exp(-f·T)
        """
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p_model = float(model.zcb_price(state, 0.0, mat))
            p_market = float(jnp.exp(-FLAT_RATE * mat))
            assert p_model == pytest.approx(p_market, rel=1e-4)

    def test_zcb_in_unit_interval(self, model: G1PP) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: G1PP) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_spot_rate_near_flat(self, model: G1PP) -> None:
        """Spot rate should be near the flat forward rate at t=0."""
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) == pytest.approx(FLAT_RATE, rel=0.05)

    def test_forward_rate_near_flat(self, model: G1PP) -> None:
        """Forward rate should be near the flat market rate at t=0."""
        state = model.init_state()
        f = model.forward_rate(state, 0.0, 5.0)
        assert float(f) == pytest.approx(FLAT_RATE, rel=0.05)

    def test_swap_rate_positive(self, model: G1PP) -> None:
        """Swap rate should be positive."""
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0

    def test_swap_rate_near_flat(self, model: G1PP) -> None:
        """With flat curve, swap rate ≈ spot rate ≈ flat_rate."""
        state = model.init_state()
        s_val = float(model.swap_rate(state, 0.0, 5.0, 1.0))
        assert s_val == pytest.approx(FLAT_RATE, rel=0.1)


class TestG1PPStochasticProcess:
    """Tests for analytic_a and analytic_b."""

    def test_analytic_b_at_zero(self, model: G1PP) -> None:
        """B(0) = 0."""
        b_val = model.analytic_b(0.0)
        assert float(b_val) == pytest.approx(0.0, abs=1e-12)

    def test_analytic_b_positive(self, model: G1PP) -> None:
        """B(τ) > 0 for τ > 0."""
        b_val = model.analytic_b(5.0)
        assert float(b_val) > 0.0

    def test_zcb_at_t0_matches_ab(self, model: G1PP) -> None:
        """At t=0 with x=0: P(0,T) = A(T) · exp(-B(T)·0) = A(T)."""
        state = model.init_state()
        tau = 5.0
        p = float(model.zcb_price(state, 0.0, tau))
        a_val = float(model.analytic_a(tau))
        assert p == pytest.approx(a_val, abs=1e-10)
